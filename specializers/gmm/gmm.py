import numpy as np
from numpy.random import *
from numpy import s_
from asp.config import PlatformDetector
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as asp_module
from codepy.cgen import *
from codepy.cuda import CudaModule
import math
import sys
from imp import find_module
from os.path import join

#TODO: Change to GMMComponents
class Components(object):
    
    def __init__(self, M, D, weights = None, means = None, covars = None):
        self.M = M
        self.D = D
        self.weights = weights if weights is not None else np.empty(M, dtype=np.float32)
        self.means = means if means is not None else  np.empty(M*D, dtype=np.float32)
        self.covars = covars if covars is not None else  np.empty(M*D*D, dtype=np.float32)
        self.comp_probs = np.empty(M, dtype=np.float32)
        
        def init_random_weights(self):
            self.weights = numpy.random.random((self.M))
            
        def init_random_means(self):
            self.means = numpy.random.random((self.M,self.D))
    
        def init_random_covars(self):
            self.covars = numpy.random.random((self.M, self.D, self.D))

        def shrink_components(self, new_M):
            self.weights = np.resize(self.weights, new_M)
            self.means = np.resize(self.means, new_M*self.D)
            self.covars = np.resize(self.covars, new_M*self.D*self.D)

            
class EvalData(object):

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.memberships = np.zeros((M,N), dtype=np.float32)
        self.loglikelihoods = np.zeros(N, dtype=np.float32)
        self.likelihood = 0.0

    def resize(self, N, M):
        self.memberships.resize((M,N))
        self.memberships = np.ascontiguousarray(self.memberships)
        self.loglikelihoods.resize(N, refcheck=False)
        self.loglikelihoods = np.ascontiguousarray(self.loglikelihoods)
        self.M = M
        self.N = N

class GMM(object):
    #Module for checking compilers and platform features.
    #TODO: We track the device id separately here because this specializer only supports using one CUDA device for all GMM instances.
    platform = PlatformDetector()
    cuda_device_id = None
    #Singleton ASP module shared by all instances of GMM
    asp_mod = None    
    def get_asp_mod(self): return GMM.asp_mod or self.initialize_asp_mod()

    #Default parameter space for code variants
    variant_param_default = { 'c++': {'dummy': ['1']},
        'cuda': {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['122'],
            'max_num_dimensions_covar_v3': ['41'],
            'max_num_components_covar_v3': ['81'],
            'covar_version_name': ['V1'] },
        'cilk': {'dummy': ['1']}
    }
    variant_param_autotune = { 'c++': {'dummy': ['1']},
        'cuda': {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['32','128','256'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['122'],
            'max_num_dimensions_covar_v3': ['41'],
            'max_num_components_covar_v3': ['81'],
            'covar_version_name': ['V1','V2A','V2B','V3'] },
        'cilk': {'dummy': ['1']}
    }

    def cuda_compilable_limits(param_dict, gpu_info):
            tpb = int(gpu_info['max_threads_per_block'])
            shmem = int(gpu_info['max_shared_memory_per_block'])
            gpumem = int(gpu_info['total_mem'])
            vname = param_dict['covar_version_name']
            eblocks = int(param_dict['num_blocks_estep'])
            ethreads = int(param_dict['num_threads_estep'])
            mthreads = int(param_dict['num_threads_mstep'])
            blocking = int(param_dict['num_event_blocks'])
            max_d = int(param_dict['max_num_dimensions'])
            max_d_v3 = int(param_dict['max_num_dimensions_covar_v3'])
            max_m = int(param_dict['max_num_components'])
            max_m_v3 = int(param_dict['max_num_components_covar_v3'])
            max_n = gpumem / (max_d*4)
            max_arg_values = (max_m, max_d, max_n)

            compilable = False

            if ethreads <= tpb and mthreads <= tpb and (max_d*max_d+max_d)*4 < shmem and ethreads*4 < shmem and mthreads*4 < shmem: 
                if vname.upper() == 'V1':
                    if (max_d + mthreads)*4 < shmem:
                        compilable = True
                elif vname.upper() == 'V2A':
                    if max_d*4 < shmem:
                        compilable = True
                elif vname.upper() == 'V2B':
                    if (max_d*max_d+max_d)*4 < shmem:
                        compilable = True
                else:
                    if (max_d_v3*max_m_v3 + mthreads + max_m_v3)*4 < shmem:
                        compilable = True
            return compilable

    def cuda_runable_limits(param_dict, gpu_info):
            tpb = int(gpu_info['max_threads_per_block'])
            shmem = int(gpu_info['max_shared_memory_per_block'])
            gpumem = int(gpu_info['total_mem'])
            vname = param_dict['covar_version_name']
            ethreads = int(param_dict['num_threads_estep'])
            mthreads = int(param_dict['num_threads_mstep'])
            blocking = int(param_dict['num_event_blocks'])
            max_d = int(param_dict['max_num_dimensions'])
            max_d_v3 = int(param_dict['max_num_dimensions_covar_v3'])
            max_m = int(param_dict['max_num_components'])
            max_m_v3 = int(param_dict['max_num_components_covar_v3'])
            max_n = gpumem / (max_d*4)
            max_arg_values = (max_m, max_d, max_n)

            def check_func(*args, **kwargs):
                if ethreads <= tpb and mthreads <= tpb and (max_d*max_d+max_d)*4 < shmem and ethreads*4 < shmem and mthreads*4 < shmem: 
                    if vname.upper() == 'V1':
                        if (max_d + mthreads)*4 < shmem:
                            return all([(a <= b) for a,b in zip(args, max_arg_values)])
                    if vname.upper() == 'V2A':
                        if max_d*4 < shmem:
                            return all([(a <= b) for a,b in zip(args, max_arg_values)]) and args[1]*(args[1]-1)/2 < tpb
                    if vname.upper() == 'V2B':
                        if (max_d*max_d+max_d)*4 < shmem:
                            return all([(a <= b) for a,b in zip(args, max_arg_values)]) and args[1]*(args[1]-1)/2 < tpb
                    else:
                        if (max_d_v3*max_m_v3 + mthreads + max_m_v3)*4 < shmem:
                            return all([(a <= b) for a,b in zip(args, (max_m_v3, max_d_v3, max_n))])
                return false
            return check_func

    def cuda_backend_render_func(self, param_dict, vals):
        param_dict['supports_float32_atomic_add'] = self.cuda_info['supports_float32_atomic_add']
        cu_kern_tpl = AspTemplate.Template(filename="templates/em_cuda_kernels.mako")
        cu_kern_rend = cu_kern_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_module([Line(cu_kern_rend)],'cuda')
        c_decl_tpl = AspTemplate.Template(filename="templates/em_cuda_launch_decl.mako") 
        c_decl_rend  = c_decl_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_preamble(c_decl_rend,'c++') #TODO: <4.1 hack
        
    def cilk_backend_render_func(self, param_dict, vals):
        cilk_kern_tpl = AspTemplate.Template(filename="templates/em_cilk_kernels.mako")
        cilk_kern_rend = cilk_kern_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_module([Line(cilk_kern_rend)],'cilk')
        c_decl_tpl = AspTemplate.Template(filename="templates/em_cilk_kernel_decl.mako") 
        c_decl_rend  = c_decl_tpl.render( param_val_list = vals, **param_dict)
        #GMM.asp_mod.add_to_preamble(c_decl_rend,'cilk')

    backend_compilable_limit_funcs = { 
        'c++':  lambda param_dict, device: True,
        'cilk': lambda param_dict, device: True,
        'cuda': cuda_compilable_limits
    }

    backend_runable_limit_funcs = { 
        'c++':  lambda param_dict, device: lambda *args, **kwargs: True,
        'cilk': lambda param_dict, device: lambda *args, **kwargs: True,
        'cuda': cuda_runable_limits
    }

    backend_specific_render_funcs = {
        'c++': lambda param_dict, vals: None,
        'cilk': cilk_backend_render_func,
        'cuda': cuda_backend_render_func
    }

    #Flags to keep track of memory allocations, singletons
    event_data_gpu_copy = None
    event_data_cpu_copy = None
    component_data_gpu_copy = None
    component_data_cpu_copy = None
    eval_data_gpu_copy = None
    eval_data_cpu_copy = None
    index_list_data_gpu_copy = None
    index_list_data_cpu_copy = None
    log_table_allocated = None
    
    # internal functions to allocate and deallocate component and event data on the CPU and GPU
    def internal_alloc_event_data(self, X):
        #if not np.array_equal(GMM.event_data_cpu_copy, X) and X is not None:
        if GMM.event_data_cpu_copy is not None:
            self.internal_free_event_data()
        self.get_asp_mod().alloc_events_on_CPU(X)
        GMM.event_data_cpu_copy = X
        if self.use_cuda:
            self.get_asp_mod().alloc_events_on_GPU(X.shape[0], X.shape[1])
            self.get_asp_mod().copy_event_data_CPU_to_GPU(X.shape[0], X.shape[1])
            GMM.event_data_gpu_copy = X

    def internal_free_event_data(self):
        if self.event_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_events_on_CPU()
            GMM.event_data_cpu_copy = None
        if GMM.event_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_events_on_GPU()
            GMM.event_data_gpu_copy = None

    def internal_alloc_event_data_from_index(self, X, I):
        #if not np.array_equal(GMM.event_data_gpu_copy, X) and X is not None:
        if GMM.event_data_gpu_copy is not None:
            self.internal_free_event_data()
        self.get_asp_mod().alloc_events_from_index_on_CPU(X, I, I.shape[0], X.shape[1])
        self.get_asp_mod().alloc_events_from_index_on_GPU(I.shape[0], X.shape[1])
        self.get_asp_mod().copy_events_from_index_CPU_to_GPU(I.shape[0], X.shape[1])
        GMM.event_data_gpu_copy = X
        GMM.event_data_cpu_copy = X
                                                                    
            
    # allocate index list for accessing subset of events
    def internal_alloc_index_list_data(self, X):
        if not np.array_equal(GMM.index_list_data_gpu_copy, X) and X is not None:
            if GMM.index_list_data_gpu_copy is not None:
                self.internal_free_index_list_data()
            self.get_asp_mod().alloc_index_list_on_CPU(X)
            self.get_asp_mod().alloc_index_list_on_GPU(X.shape[0])
            self.get_asp_mod().copy_index_list_data_CPU_to_GPU(X.shape[0])
            GMM.index_list_data_gpu_copy = X
            GMM.index_list_data_cpu_copy = X
                
    def internal_free_index_list_data(self):
        if GMM.index_list_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_index_list_on_GPU()
            GMM.index_list_data_gpu_copy = None
        if self.index_list_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_index_list_on_CPU()
            GMM.index_list_data_cpu_copy = None
                
            
    def internal_alloc_component_data(self):
        if GMM.component_data_cpu_copy != self.components:
            if GMM.component_data_cpu_copy:
                self.internal_free_component_data()
            self.get_asp_mod().alloc_components_on_CPU(self.M, self.D, self.components.weights, self.components.means, self.components.covars, self.components.comp_probs)
            GMM.component_data_cpu_copy = self.components
            if self.use_cuda:
                self.get_asp_mod().alloc_components_on_GPU(self.M, self.D)
                self.get_asp_mod().copy_component_data_CPU_to_GPU(self.M, self.D)
                GMM.component_data_gpu_copy = self.components
            
    def internal_free_component_data(self):
        if GMM.component_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_components_on_CPU()
            GMM.component_data_cpu_copy = None
        if GMM.component_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_components_on_GPU()
            GMM.component_data_gpu_copy = None

    def internal_alloc_eval_data(self, X):
        if X is not None:
            if self.eval_data.M != self.M or self.eval_data.N != X.shape[0] or GMM.eval_data_cpu_copy != self.eval_data:
                if GMM.eval_data_cpu_copy is not None:
                    self.internal_free_eval_data()
                self.eval_data.resize(X.shape[0], self.M)
                self.get_asp_mod().alloc_evals_on_CPU(self.eval_data.memberships, self.eval_data.loglikelihoods)
#                self.get_asp_mod().alloc_evals_on_CPU(self.eval_data.memberships, self.eval_data.loglikelihoods)
                GMM.eval_data_cpu_copy = self.eval_data
                if self.use_cuda:
                    self.get_asp_mod().alloc_evals_on_GPU(X.shape[0], self.M)
                    GMM.eval_data_gpu_copy = self.eval_data

    def internal_alloc_eval_data_from_index(self, X, length):
        if X is not None:
            if GMM.eval_data_gpu_copy is not None:
                self.internal_free_eval_data()
            self.eval_data.resize(X.shape[0], self.M)
            self.get_asp_mod().alloc_evals_on_GPU(length, self.M)
            self.get_asp_mod().alloc_evals_on_CPU(self.eval_data.memberships, self.eval_data.loglikelihoods)
            GMM.eval_data_gpu_copy = self.eval_data
            GMM.eval_data_cpu_copy = self.eval_data

    def internal_free_eval_data(self):
        if GMM.eval_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_evals_on_CPU()
            GMM.eval_data_cpu_copy = None
        if GMM.eval_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_evals_on_GPU()
            GMM.eval_data_gpu_copy = None

    def internal_seed_data(self, X, D, N):
        print self.components.weights                                                                    
        print self.components.means                                                                      
        print self.components.covars
        print self.components.comp_probs
        getattr(self.get_asp_mod(),'seed_components_'+self.cvtype)(self.M, D, N)
        self.components_seeded = True
        self.get_asp_mod().copy_component_data_GPU_to_CPU(self.M, D)
        print self.components.weights                                                                    
        print self.components.means                                                                      
        print self.components.covars
        print self.components.comp_probs


<<<<<<< HEAD

    def __init__(self, M, D, means=None, covars=None, weights=None, cvtype='diag', names_of_backends_to_use=['cuda'], autotune=False, device_id=0): #TODO: Make default backend 'base'
=======
    def __init__(self, M, D, means=None, covars=None, weights=None, cvtype=1, names_of_backends_to_use=['cuda'], variant_param_spaces=None, device_id=0): #TODO: Make default backend 'base'
>>>>>>> Changes to make Cilk backend work on ASP 0.1.2, requires Cilk V12.0.5
        self.M = M
        self.D = D
        self.cvtype = cvtype

        self.variant_param_spaces = GMM.variant_param_autotune if autotune else GMM.variant_param_defaults
        self.names_of_backends_to_use = names_of_backends_to_use
        self.components = Components(M, D, weights, means, covars)
        self.eval_data = EvalData(1, M)
        self.platform_info = {}
        self.clf = None # pure python mirror module
        self.use_cuda = False
        self.use_cilk = False
        if 'cuda' in names_of_backends_to_use:
            if 'nvcc' in GMM.platform.get_compilers() and GMM.platform.get_num_cuda_devices() > 0:
                self.use_cuda = True
                self.platform_info['cuda'] = GMM.platform.get_cuda_info()
                if GMM.cuda_device_id == None:
                    GMM.cuda_device_id = device_id
                    GMM.platform.set_cuda_device(device_id)
                elif GMM.cuda_device_id != device_id:
                    #TODO: May actually be allowable if deallocate all GPU allocations first?
                    print "WARNING: As python only has one thread context, it can only use one GPU at a time, and you are attempting to run on a second GPU."
            else: print "WARNING: You asked for a CUDA backend but no compiler was found."
        if 'cilk' in names_of_backends_to_use:

            if 'icc' in GMM.platform.get_compilers():
                self.use_cilk = True
                self.platform_info['cilk'] = GMM.platform.get_cpu_info()
            else: print "WARNING: You asked for a Cilk backend but no compiler was found."

        if means is None and covars is None and weights is None:
            self.components_seeded = False
        else:
            self.components_seeded = True

            
    #Called the first time a GMM instance tries to use a specialized function
    def initialize_asp_mod(self):
        # Create ASP module
        GMM.asp_mod = asp_module.ASPModule(use_cuda=self.use_cuda, use_cilk=self.use_cilk)

        if self.use_cuda:
            self.insert_base_code_into_listed_modules(['c++'])
            self.insert_non_rendered_code_into_cuda_module()

            self.insert_rendered_code_into_module('cuda')
            #GMM.asp_mod.backends['c++'].toolchain.cc = 'gcc'
            #GMM.asp_mod.backends['c++'].toolchain.cflags.append('-fPIC')
            GMM.asp_mod.backends['cuda'].toolchain.cflags.extend(["-Xcompiler","-fPIC","-arch=sm_%s%s" % self.platform_info['cuda']['capability'] ])
            #GMM.asp_mod.backends['cuda'].toolchain.add_library("project",['.','./include'],[],[])  
            GMM.asp_mod.backends['c++'].compilable = False # TODO: For now, must use cuda backend to compile

        if self.use_cilk:
            self.insert_base_code_into_listed_modules(['cilk'])
            self.insert_non_rendered_code_into_cilk_module()
            self.insert_rendered_code_into_module('cilk')
            GMM.asp_mod.backends['cilk'].toolchain.cc = 'icc'
            GMM.asp_mod.backends['cilk'].toolchain.cflags = ['-O2','-gcc', '-ip','-fPIC']

        # Setup toolchain and compile
	from codepy.libraries import add_numpy, add_boost_python, add_cuda
        for name, mod in GMM.asp_mod.backends.iteritems():
            add_numpy(mod.toolchain)
            add_boost_python(mod.toolchain)
            if name in ['cuda']:
                add_cuda(mod.toolchain) #TODO: for now, need to add cuda to c++ toolchain as it might contain host-side cuda funcs
        return GMM.asp_mod

    def insert_base_code_into_listed_modules(self, names_of_backends):
        c_base_tpl = AspTemplate.Template(filename="templates/em_base_helper_funcs.mako")
        c_base_rend = c_base_tpl.render()
        component_t_decl ="""
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* CP; //cluster probability [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""

        base_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h', 'numpy/arrayobject.h']
        for b_name in names_of_backends:
            for header in base_system_header_names: 
                GMM.asp_mod.add_to_preamble([Include(header, True)], b_name)
            #Add Boost interface links for components and distance objects
            GMM.asp_mod.add_to_init("import_array();", b_name)
            GMM.asp_mod.add_to_init("""boost::python::class_<components_struct>("Components");
                boost::python::scope().attr("components") = boost::python::object(boost::python::ptr(&components));""", b_name)
            GMM.asp_mod.add_to_init("""
                 boost::python::class_<return_component_container>("ReturnClusterContainer")
                 .def_readwrite("new_component", &return_component_container::component)
                 .def_readwrite("distance", &return_component_container::distance);
                 boost::python::scope().attr("component_distance") = boost::python::object(boost::python::ptr(&ret));""", b_name)
            GMM.asp_mod.add_to_module([Line(c_base_rend)],b_name)
            GMM.asp_mod.add_to_preamble(component_t_decl, b_name)

    def insert_non_rendered_code_into_cuda_module(self):
        #Add C/CUDA source code that is not based on code variant parameters

        #Add decls to preamble necessary for linking to compiled CUDA sources
        component_t_decl =""" 
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* CP; //cluster probability [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""
        GMM.asp_mod.add_to_preamble(component_t_decl,'cuda')

        #TODO: Move this back into insert_base_code_into_listed_modules for cuda 4.1
        names_of_helper_funcs = ["alloc_events_on_CPU", "alloc_components_on_CPU", "alloc_evals_on_CPU", "dealloc_events_on_CPU", "dealloc_components_on_CPU", "dealloc_temp_components_on_CPU", "dealloc_evals_on_CPU", "relink_components_on_CPU", "compute_distance_rissanen", "merge_components", "create_lut_log_table", "compute_KL_distance"]
        for fname in names_of_helper_funcs:
            GMM.asp_mod.add_helper_function(fname, "", 'cuda')

        #Add bodies of helper functions
        c_base_tpl = AspTemplate.Template(filename="templates/em_cuda_host_helper_funcs.mako")
        c_base_rend  = c_base_tpl.render()
        GMM.asp_mod.add_to_module([Line(c_base_rend)],'c++')
        cu_base_tpl = AspTemplate.Template(filename="templates/em_cuda_device_helper_funcs.mako")
        cu_base_rend = cu_base_tpl.render()
        GMM.asp_mod.add_to_module([Line(cu_base_rend)],'cuda')
        #Add Boost interface links for helper functions
        names_of_cuda_helper_funcs = ["alloc_events_on_GPU","alloc_index_list_on_GPU", "alloc_events_from_index_on_GPU", "alloc_components_on_GPU","alloc_evals_on_GPU","copy_event_data_CPU_to_GPU", "copy_index_list_data_CPU_to_GPU", "copy_events_from_index_CPU_to_GPU", "copy_component_data_CPU_to_GPU", "copy_component_data_GPU_to_CPU", "copy_evals_CPU_to_GPU", "copy_evals_data_GPU_to_CPU","dealloc_events_on_GPU","dealloc_components_on_GPU", "dealloc_evals_on_GPU", "dealloc_index_list_on_GPU"] 
        for fname in names_of_cuda_helper_funcs:
            GMM.asp_mod.add_helper_function(fname,"",'cuda')

    def insert_non_rendered_code_into_cilk_module(self):
        component_t_decl =""" 
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* CP; //cluster probability [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""
        #GMM.asp_mod.add_to_preamble(component_t_decl,'cilk')

        #TODO: Move this back into insert_base_code_into_listed_modules for cuda 4.1
        names_of_helper_funcs = ["alloc_events_on_CPU", "alloc_components_on_CPU", "alloc_evals_on_CPU", "dealloc_events_on_CPU", "dealloc_components_on_CPU", "dealloc_temp_components_on_CPU", "dealloc_evals_on_CPU", "relink_components_on_CPU", "compute_distance_rissanen", "merge_components", "create_lut_log_table", "compute_KL_distance"]
        for fname in names_of_helper_funcs:
            GMM.asp_mod.add_helper_function(fname, "", 'cilk')

        cilk_base_tpl = AspTemplate.Template(filename="templates/em_cilk_helper_funcs.mako")
        cilk_base_rend = cilk_base_tpl.render()
        #GMM.asp_mod.add_to_module([Line(cilk_base_rend)],'cilk')

        #Add Cilk source code that is not based on code variant parameters
        system_header_names = ['cilk/cilk.h','cilk/reducer_opadd.h']  
        for x in system_header_names: 
            GMM.asp_mod.add_to_preamble([Include(x, True)],'cilk')

    def render_func_variant( self, param_dict, param_val_list, can_be_compiled, backend_name, func_name):
        def var_name_generator(base):
            return '_'.join(['em',backend_name,base]+param_val_list)
        if can_be_compiled: 

            c_tpl = AspTemplate.Template(filename='_'.join(["templates/em",backend_name,func_name+".mako"]))
            func_body = c_tpl.render( param_val_list = param_val_list, **param_dict)
        else:
            func_body = "void " + var_name_generator(func_name) + "(int m, int d, int n, PyObject *data){}"

        return var_name_generator(func_name), func_body

    def generate_permutations (self, key_arr, val_arr_arr, current, compilable, make_run_check, backend_specific_render_func, backend_name, func_names, cvtype, result):
        idx = len(current)
        name = key_arr[idx]
        for v in val_arr_arr[idx]:
            current[name]  = v
            if idx == len(key_arr)-1:
                # Get vals based on alphabetical order of keys
                param_dict = current.copy()
                param_dict['diag_only'] = '1' if cvtype == 'diag' else '0'
                param_names = param_dict.keys()
                param_names.sort()
                vals = map(param_dict.get, param_names)
                # Use vals to render templates 
                can_be_compiled = compilable(param_dict, self.platform_info[backend_name])
                if can_be_compiled:
                    backend_specific_render_func(self, param_dict, vals)
                run_check = make_run_check(param_dict, self.platform_info[backend_name]) 
                for func_name in func_names:
                    v_name, v_body = self.render_func_variant(param_dict, vals, can_be_compiled, backend_name, func_name)
                    result.setdefault(func_name,[]).append((v_name,v_body,run_check))
            else:
                self.generate_permutations(key_arr, val_arr_arr, current, compilable, make_run_check, backend_specific_render_func, backend_name, func_names, cvtype, result)
        del current[name]

    def insert_rendered_code_into_module(self, backend_name):
        import hashlib
        key_func = lambda *args, **kwargs: hashlib.md5(str([args[0],args[1],math.floor(math.log10(args[2]))])+str(kwargs)).hexdigest()
        for cvtype in ['diag', 'full']:
            func_names = ['train', 'eval', 'seed_components']
            all_variants = {}
            self.generate_permutations( self.variant_param_spaces[backend_name].keys(),
                                        self.variant_param_spaces[backend_name].values(), {}, 
                                        GMM.backend_compilable_limit_funcs[backend_name], 
                                        GMM.backend_runable_limit_funcs[backend_name],
                                        GMM.backend_specific_render_funcs[backend_name], 
                                        backend_name, func_names, cvtype, all_variants)
            for func_name in func_names:
                names  = [r[0] for r in all_variants[func_name]]
                bodies = [r[1] for r in all_variants[func_name]]
                checks = [r[2] for r in all_variants[func_name]]
                GMM.asp_mod.add_function(   '_'.join([func_name, cvtype]), 
                                            bodies, 
                                            variant_names = names,
                                            run_check_funcs = checks,
                                            key_function = key_func,
                                            backend = backend_name)

    def __del__(self):
        self.internal_free_event_data()
        self.internal_free_component_data()
        self.internal_free_eval_data()
    
    def train_using_python(self, input_data, iters=10):
        from sklearn import mixture
        self.clf = mixture.GMM(n_components=self.M, cvtype=self.cvtype)
        self.clf.fit(input_data, n_iter=iters)
        return self.clf.means, self.clf.covars
    
    def eval_using_python(self, obs_data):
        from sklearn import mixture
        if self.clf is not None:
            return self.clf.eval(obs_data)
        else: return []

    def predict_using_python(self, obs_data):
        from sklearn import mixture
        if self.clf is not None:
            return self.clf.predict(obs_data)
        else: return []

    def train(self, input_data, min_em_iters=1, max_em_iters=10):
        N = input_data.shape[0] 
        if input_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (input_data.shape[1], self.D)
        self.internal_alloc_event_data(input_data)
        self.internal_alloc_eval_data(input_data)
        self.internal_alloc_component_data()
        
        if not self.components_seeded:
            self.internal_seed_data(input_data, input_data.shape[1], input_data.shape[0])

        self.eval_data.likelihood = getattr(self.get_asp_mod(),'train_'+self.cvtype)(self.M, self.D, N, min_em_iters, max_em_iters)[0]

        self.components.means = self.components.means.reshape(self.M, self.D)
        self.components.covars = self.components.covars.reshape(self.M, self.D, self.D)
        
        return self


        #TODO: expose only one function to the domain programmer
        #handle selection of gather mechanisms internally

        #train on subset
        #collect indices in python
        #gather in CUDA
    def train_on_subset(self, input_data, index_list):
        N = input_data.shape[0]
        K = index_list.shape[0] #number of indices
        
        if input_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (input_data.shape[1], self.D)
        self.internal_alloc_event_data(input_data)
        self.internal_alloc_index_list_data(index_list)
        self.internal_alloc_eval_data(input_data)
        self.internal_alloc_component_data()
        
        if not self.components_seeded:
            self.internal_seed_data(input_data, input_data.shape[1], input_data.shape[0])
            
        self.eval_data.likelihood = self.get_asp_mod().train_on_subset(self.M, self.D, N, K)[0]
        return self
        
        
        #train on subset
        #collect indices in python
        #gather in C
    def train_on_subset_c(self, input_data, index_list):
        N = input_data.shape[0]
        K = index_list.shape[0] #number of indices
        
        if input_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (input_data.shape[1], self.D)
            
        self.internal_alloc_event_data_from_index(input_data, index_list)
        self.internal_alloc_eval_data(input_data)
        self.internal_alloc_component_data()
            
        if not self.components_seeded:
            self.internal_seed_data(input_data, input_data.shape[1], K)

        self.eval_data.likelihood = self.get_asp_mod().train(self.M, self.D, K)[0]
        return self
            
    
    def eval(self, obs_data):
        N = obs_data.shape[0]
        if obs_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (obs_data.shape[1], self.D)
        self.internal_alloc_event_data(obs_data)
        self.internal_alloc_eval_data(obs_data)
        self.internal_alloc_component_data()

        self.eval_data.likelihood = getattr(self.get_asp_mod(),'eval_'+self.cvtype)(self.M, self.D, N)

        logprob = self.eval_data.loglikelihoods
        posteriors = self.eval_data.memberships
        return logprob, posteriors # N log probabilities, NxM posterior probabilities for each component

    def score(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob # N log probabilities

    def decode(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob, posteriors.argmax(axis=0) # N log probabilities, N indexes of most likely components 

    def predict(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return posteriors.argmax(axis=0) # N indexes of most likely components

    def merge_components(self, c1, c2, new_component):
        self.get_asp_mod().dealloc_temp_components_on_CPU()
        self.get_asp_mod().merge_components(c1, c2, new_component, self.M, self.D)
        self.M -= 1
        self.components.shrink_components(self.M)
        self.get_asp_mod().relink_components_on_CPU(self.components.weights, self.components.means, self.components.covars)

    def compute_distance_rissanen(self, c1, c2):
        self.get_asp_mod().compute_distance_rissanen(c1, c2, self.D)
        new_component = self.get_asp_mod().compiled_module.component_distance.new_component
        dist = self.get_asp_mod().compiled_module.component_distance.distance
        return new_component, dist


    def find_top_KL_pairs(self, K, gmm_list):
        if GMM.log_table_allocated is None:
            self.get_asp_mod().create_lut_log_table()
            GMM.log_table_allocated = 1
            
        l = len(gmm_list)
        score_list = []
        for gmm1idx in range(l):
            for gmm2idx in range(gmm1idx+1, l):
                score = self.get_asp_mod().compute_KL_distance(gmm_list[gmm1idx].D, gmm_list[gmm1idx].M, gmm_list[gmm2idx].M, gmm_list[gmm1idx].components.weights, gmm_list[gmm1idx].components.means, gmm_list[gmm1idx].components.covars,gmm_list[gmm1idx].components.comp_probs, gmm_list[gmm2idx].components.weights, gmm_list[gmm2idx].components.means, gmm_list[gmm2idx].components.covars, gmm_list[gmm2idx].components.comp_probs)
                score_list.append((score, (gmm1idx,gmm2idx)))

        sorted_list = sorted(score_list, key=lambda score: score[0])
        ret_list = []
        if K==-1: #all
            for k in range(len(sorted_list)):
                ret_list.append(sorted_list[k][1])
        else:
            if(len(gmm_list)>=K):
                for k in range(0,K):
                    ret_list.append(sorted_list[k][1])
            else:
                for k in range(0,len(gmm_list)-1):
                    ret_list.append(sorted_list[k][1])
        return ret_list
                            

def compute_distance_BIC(gmm1, gmm2, data, em_iters=10):
    cd1_M = gmm1.M
    cd2_M = gmm2.M
    nComps = cd1_M + cd2_M

    ratio1 = float(cd1_M)/float(nComps)
    ratio2 = float(cd2_M)/float(nComps)

    w = np.append(ratio1*gmm1.components.weights, ratio2*gmm2.components.weights)
    m = np.append(gmm1.components.means, gmm2.components.means)
    c = np.append(gmm1.components.covars, gmm2.components.covars)

    temp_GMM = GMM(nComps, gmm1.D, weights=w, means=m, covars=c, cvtype=gmm1.cvtype, names_of_backends_to_use=gmm1.names_of_backends_to_use, variant_param_spaces=gmm1.variant_param_spaces)

    temp_GMM.train(data, max_em_iters=em_iters)
    score = temp_GMM.eval_data.likelihood - (gmm1.eval_data.likelihood + gmm2.eval_data.likelihood)
    return temp_GMM, score

def compute_distance_BIC_idx(gmm1, gmm2, data, index_list):
    cd1_M = gmm1.M
    cd2_M = gmm2.M
    nComps = cd1_M + cd2_M
    
    ratio1 = float(cd1_M)/float(nComps)
    ratio2 = float(cd2_M)/float(nComps)
    
    w = np.append(ratio1*gmm1.components.weights, ratio2*gmm2.components.weights)
    m = np.append(gmm1.components.means, gmm2.components.means)
    c = np.append(gmm1.components.covars, gmm2.components.covars)
    temp_GMM = GMM(nComps, gmm1.D, weights=w, means=m, covars=c, names_of_backends_to_use=gmm1.names_of_backends_to_use, variant_param_spaces=gmm1.variant_param_spaces, device_id=gmm1.device_id)
    
    temp_GMM.train_on_subset_c(data, index_list)
    
    score = temp_GMM.eval_data.likelihood - (gmm1.eval_data.likelihood + gmm2.eval_data.likelihood)
    
    return temp_GMM, score


