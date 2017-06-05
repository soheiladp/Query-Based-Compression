%module cutil

%{
#define SWIG_FILE_WITH_INIT
#include "cutil.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (int* IN_ARRAY1, int DIM1) {(int* x, int n), (int* weights, int w)};
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* hist, int n1)};
%include "cutil.h"

%clear (int* x, int n), (int* weights, int w);
