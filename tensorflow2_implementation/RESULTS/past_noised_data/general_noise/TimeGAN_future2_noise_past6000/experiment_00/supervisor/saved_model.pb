јЛ(
ЭЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18Нм&
p

OUT/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
OUT/kernel
i
OUT/kernel/Read/ReadVariableOpReadVariableOp
OUT/kernel*
_output_shapes

:*
dtype0
h
OUT/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
OUT/bias
a
OUT/bias/Read/ReadVariableOpReadVariableOpOUT/bias*
_output_shapes
:*
dtype0

GRU_1/gru_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameGRU_1/gru_cell_26/kernel

,GRU_1/gru_cell_26/kernel/Read/ReadVariableOpReadVariableOpGRU_1/gru_cell_26/kernel*
_output_shapes

:<*
dtype0
 
"GRU_1/gru_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*3
shared_name$"GRU_1/gru_cell_26/recurrent_kernel

6GRU_1/gru_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp"GRU_1/gru_cell_26/recurrent_kernel*
_output_shapes

:<*
dtype0

GRU_1/gru_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameGRU_1/gru_cell_26/bias

*GRU_1/gru_cell_26/bias/Read/ReadVariableOpReadVariableOpGRU_1/gru_cell_26/bias*
_output_shapes

:<*
dtype0

GRU_2/gru_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameGRU_2/gru_cell_27/kernel

,GRU_2/gru_cell_27/kernel/Read/ReadVariableOpReadVariableOpGRU_2/gru_cell_27/kernel*
_output_shapes

:<*
dtype0
 
"GRU_2/gru_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*3
shared_name$"GRU_2/gru_cell_27/recurrent_kernel

6GRU_2/gru_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp"GRU_2/gru_cell_27/recurrent_kernel*
_output_shapes

:<*
dtype0

GRU_2/gru_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameGRU_2/gru_cell_27/bias

*GRU_2/gru_cell_27/bias/Read/ReadVariableOpReadVariableOpGRU_2/gru_cell_27/bias*
_output_shapes

:<*
dtype0

NoOpNoOp
Є
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*п
valueеBв BЫ
з
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api

signatures

	cell

_inbound_nodes

state_spec
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api

cell
_inbound_nodes

state_spec
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
|
_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
8
 0
!1
"2
#3
$4
%5
6
7
8
 0
!1
"2
#3
$4
%5
6
7
 
­
&non_trainable_variables
'layer_regularization_losses
	variables
trainable_variables
(layer_metrics
regularization_losses

)layers
*metrics
 
~

 kernel
!recurrent_kernel
"bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
 
 
 

 0
!1
"2

 0
!1
"2
 
Й
/non_trainable_variables
0layer_regularization_losses
	variables
trainable_variables
1layer_metrics
regularization_losses

2states

3layers
4metrics
~

#kernel
$recurrent_kernel
%bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
 
 
 

#0
$1
%2

#0
$1
%2
 
Й
9non_trainable_variables
:layer_regularization_losses
	variables
trainable_variables
;layer_metrics
regularization_losses

<states

=layers
>metrics
 
VT
VARIABLE_VALUE
OUT/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEOUT/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
?non_trainable_variables
@layer_regularization_losses
	variables
trainable_variables
Alayer_metrics
regularization_losses

Blayers
Cmetrics
TR
VARIABLE_VALUEGRU_1/gru_cell_26/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"GRU_1/gru_cell_26/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEGRU_1/gru_cell_26/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEGRU_2/gru_cell_27/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"GRU_2/gru_cell_27/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEGRU_2/gru_cell_27/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
 

 0
!1
"2

 0
!1
"2
 
­
Dnon_trainable_variables
Elayer_regularization_losses
+	variables
,trainable_variables
Flayer_metrics
-regularization_losses

Glayers
Hmetrics
 
 
 
 

	0
 

#0
$1
%2

#0
$1
%2
 
­
Inon_trainable_variables
Jlayer_regularization_losses
5	variables
6trainable_variables
Klayer_metrics
7regularization_losses

Llayers
Mmetrics
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_GRU_1_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_GRU_1_inputGRU_1/gru_cell_26/biasGRU_1/gru_cell_26/kernel"GRU_1/gru_cell_26/recurrent_kernelGRU_2/gru_cell_27/biasGRU_2/gru_cell_27/kernel"GRU_2/gru_cell_27/recurrent_kernel
OUT/kernelOUT/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_694364
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameOUT/kernel/Read/ReadVariableOpOUT/bias/Read/ReadVariableOp,GRU_1/gru_cell_26/kernel/Read/ReadVariableOp6GRU_1/gru_cell_26/recurrent_kernel/Read/ReadVariableOp*GRU_1/gru_cell_26/bias/Read/ReadVariableOp,GRU_2/gru_cell_27/kernel/Read/ReadVariableOp6GRU_2/gru_cell_27/recurrent_kernel/Read/ReadVariableOp*GRU_2/gru_cell_27/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_697475
р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
OUT/kernelOUT/biasGRU_1/gru_cell_26/kernel"GRU_1/gru_cell_26/recurrent_kernelGRU_1/gru_cell_26/biasGRU_2/gru_cell_27/kernel"GRU_2/gru_cell_27/recurrent_kernelGRU_2/gru_cell_27/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_697509ЕЃ&
щ<
ж
A__inference_GRU_2_layer_call_and_return_conditional_losses_693474

inputs
gru_cell_27_693398
gru_cell_27_693400
gru_cell_27_693402
identityЂ#gru_cell_27/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2№
#gru_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_27_693398gru_cell_27_693400gru_cell_27_693402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_6930332%
#gru_cell_27/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_27_693398gru_cell_27_693400gru_cell_27_693402*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_693410*
condR
while_cond_693409*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0$^gru_cell_27/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_27/StatefulPartitionedCall#gru_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
нH
з
GRU_2_while_body_694588(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_27_readvariableop_resource_0<
8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_27_readvariableop_resource:
6gru_2_while_gru_cell_27_matmul_readvariableop_resource<
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resourceЯ
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0gru_2_while_placeholderFGRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_2/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_2/while/gru_cell_27/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_27/ReadVariableOpВ
GRU_2/while/gru_cell_27/unstackUnpack.GRU_2/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_27/unstackз
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_27/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_27/MatMulг
GRU_2/while/gru_cell_27/BiasAddBiasAdd(GRU_2/while/gru_cell_27/MatMul:product:0(GRU_2/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_27/BiasAdd
GRU_2/while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_27/Const
'GRU_2/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_27/split/split_dim
GRU_2/while/gru_cell_27/splitSplit0GRU_2/while/gru_cell_27/split/split_dim:output:0(GRU_2/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_27/splitн
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_27/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_27/MatMul_1й
!GRU_2/while/gru_cell_27/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_27/MatMul_1:product:0(GRU_2/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_27/BiasAdd_1
GRU_2/while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_27/Const_1Ё
)GRU_2/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_27/split_1/split_dimЫ
GRU_2/while/gru_cell_27/split_1SplitV*GRU_2/while/gru_cell_27/BiasAdd_1:output:0(GRU_2/while/gru_cell_27/Const_1:output:02GRU_2/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_27/split_1Ч
GRU_2/while/gru_cell_27/addAddV2&GRU_2/while/gru_cell_27/split:output:0(GRU_2/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add 
GRU_2/while/gru_cell_27/SigmoidSigmoidGRU_2/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_27/SigmoidЫ
GRU_2/while/gru_cell_27/add_1AddV2&GRU_2/while/gru_cell_27/split:output:1(GRU_2/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_1І
!GRU_2/while/gru_cell_27/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_27/Sigmoid_1Ф
GRU_2/while/gru_cell_27/mulMul%GRU_2/while/gru_cell_27/Sigmoid_1:y:0(GRU_2/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mulТ
GRU_2/while/gru_cell_27/add_2AddV2&GRU_2/while/gru_cell_27/split:output:2GRU_2/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_2
GRU_2/while/gru_cell_27/TanhTanh!GRU_2/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/TanhЗ
GRU_2/while/gru_cell_27/mul_1Mul#GRU_2/while/gru_cell_27/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_1
GRU_2/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_27/sub/xР
GRU_2/while/gru_cell_27/subSub&GRU_2/while/gru_cell_27/sub/x:output:0#GRU_2/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/subК
GRU_2/while/gru_cell_27/mul_2MulGRU_2/while/gru_cell_27/sub:z:0 GRU_2/while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_2П
GRU_2/while/gru_cell_27/add_3AddV2!GRU_2/while/gru_cell_27/mul_1:z:0!GRU_2/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_2/while/TensorArrayV2Write/TensorListSetItemh
GRU_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add/y
GRU_2/while/addAddV2gru_2_while_placeholderGRU_2/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/addl
GRU_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add_1/y
GRU_2/while/add_1AddV2$gru_2_while_gru_2_while_loop_counterGRU_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/add_1p
GRU_2/while/IdentityIdentityGRU_2/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity
GRU_2/while/Identity_1Identity*gru_2_while_gru_2_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_2/while/Identity_1r
GRU_2/while/Identity_2IdentityGRU_2/while/add:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_2
GRU_2/while/Identity_3Identity@GRU_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_3
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resource:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_27_matmul_readvariableop_resource8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_27_readvariableop_resource1gru_2_while_gru_cell_27_readvariableop_resource_0"5
gru_2_while_identityGRU_2/while/Identity:output:0"9
gru_2_while_identity_1GRU_2/while/Identity_1:output:0"9
gru_2_while_identity_2GRU_2/while/Identity_2:output:0"9
gru_2_while_identity_3GRU_2/while/Identity_3:output:0"9
gru_2_while_identity_4GRU_2/while/Identity_4:output:0"Р
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
а
Њ
while_cond_696379
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_696379___redundant_placeholder04
0while_while_cond_696379___redundant_placeholder14
0while_while_cond_696379___redundant_placeholder24
0while_while_cond_696379___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
@
Е
while_body_693714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_26_readvariableop_resource_06
2while_gru_cell_26_matmul_readvariableop_resource_08
4while_gru_cell_26_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_26_readvariableop_resource4
0while_gru_cell_26_matmul_readvariableop_resource6
2while_gru_cell_26_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_26/ReadVariableOp 
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_26/unstackХ
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_26/MatMul/ReadVariableOpг
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMulЛ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAddt
while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_26/Const
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_26/split/split_dimє
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/splitЫ
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_26/MatMul_1/ReadVariableOpМ
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMul_1С
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAdd_1
while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_26/Const_1
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_26/split_1/split_dim­
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0"while/gru_cell_26/Const_1:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/split_1Џ
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/SigmoidГ
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_1
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Sigmoid_1Ќ
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mulЊ
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_2
while/gru_cell_26/TanhTanhwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Tanh
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_1w
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_26/sub/xЈ
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/subЂ
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_2Ї
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 


&__inference_GRU_1_layer_call_fn_696481

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_1_layer_call_and_return_conditional_losses_6936452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
щ
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_693033

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Ъс

F__inference_Supervisor_layer_call_and_return_conditional_losses_695429

inputs-
)gru_1_gru_cell_26_readvariableop_resource4
0gru_1_gru_cell_26_matmul_readvariableop_resource6
2gru_1_gru_cell_26_matmul_1_readvariableop_resource-
)gru_2_gru_cell_27_readvariableop_resource4
0gru_2_gru_cell_27_matmul_readvariableop_resource6
2gru_2_gru_cell_27_matmul_1_readvariableop_resource)
%out_tensordot_readvariableop_resource'
#out_biasadd_readvariableop_resource
identityЂGRU_1/whileЂGRU_2/whileP
GRU_1/ShapeShapeinputs*
T0*
_output_shapes
:2
GRU_1/Shape
GRU_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice/stack
GRU_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_1
GRU_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_2
GRU_1/strided_sliceStridedSliceGRU_1/Shape:output:0"GRU_1/strided_slice/stack:output:0$GRU_1/strided_slice/stack_1:output:0$GRU_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_sliceh
GRU_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/mul/y
GRU_1/zeros/mulMulGRU_1/strided_slice:output:0GRU_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/mulk
GRU_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_1/zeros/Less/y
GRU_1/zeros/LessLessGRU_1/zeros/mul:z:0GRU_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/Lessn
GRU_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/packed/1
GRU_1/zeros/packedPackGRU_1/strided_slice:output:0GRU_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_1/zeros/packedk
GRU_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/zeros/Const
GRU_1/zerosFillGRU_1/zeros/packed:output:0GRU_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/zeros
GRU_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose/perm
GRU_1/transpose	TransposeinputsGRU_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transposea
GRU_1/Shape_1ShapeGRU_1/transpose:y:0*
T0*
_output_shapes
:2
GRU_1/Shape_1
GRU_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_1/stack
GRU_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_1
GRU_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_2
GRU_1/strided_slice_1StridedSliceGRU_1/Shape_1:output:0$GRU_1/strided_slice_1/stack:output:0&GRU_1/strided_slice_1/stack_1:output:0&GRU_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_slice_1
!GRU_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/TensorArrayV2/element_shapeЪ
GRU_1/TensorArrayV2TensorListReserve*GRU_1/TensorArrayV2/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2Ы
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_1/transpose:y:0DGRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_1/TensorArrayUnstack/TensorListFromTensor
GRU_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_2/stack
GRU_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_1
GRU_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_2 
GRU_1/strided_slice_2StridedSliceGRU_1/transpose:y:0$GRU_1/strided_slice_2/stack:output:0&GRU_1/strided_slice_2/stack_1:output:0&GRU_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_2Ў
 GRU_1/gru_cell_26/ReadVariableOpReadVariableOp)gru_1_gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_26/ReadVariableOp 
GRU_1/gru_cell_26/unstackUnpack(GRU_1/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_26/unstackУ
'GRU_1/gru_cell_26/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_26/MatMul/ReadVariableOpС
GRU_1/gru_cell_26/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMulЛ
GRU_1/gru_cell_26/BiasAddBiasAdd"GRU_1/gru_cell_26/MatMul:product:0"GRU_1/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAddt
GRU_1/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_26/Const
!GRU_1/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_26/split/split_dimє
GRU_1/gru_cell_26/splitSplit*GRU_1/gru_cell_26/split/split_dim:output:0"GRU_1/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/splitЩ
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_26/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMul_1С
GRU_1/gru_cell_26/BiasAdd_1BiasAdd$GRU_1/gru_cell_26/MatMul_1:product:0"GRU_1/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAdd_1
GRU_1/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_26/Const_1
#GRU_1/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_26/split_1/split_dim­
GRU_1/gru_cell_26/split_1SplitV$GRU_1/gru_cell_26/BiasAdd_1:output:0"GRU_1/gru_cell_26/Const_1:output:0,GRU_1/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/split_1Џ
GRU_1/gru_cell_26/addAddV2 GRU_1/gru_cell_26/split:output:0"GRU_1/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add
GRU_1/gru_cell_26/SigmoidSigmoidGRU_1/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/SigmoidГ
GRU_1/gru_cell_26/add_1AddV2 GRU_1/gru_cell_26/split:output:1"GRU_1/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_1
GRU_1/gru_cell_26/Sigmoid_1SigmoidGRU_1/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Sigmoid_1Ќ
GRU_1/gru_cell_26/mulMulGRU_1/gru_cell_26/Sigmoid_1:y:0"GRU_1/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mulЊ
GRU_1/gru_cell_26/add_2AddV2 GRU_1/gru_cell_26/split:output:2GRU_1/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_2
GRU_1/gru_cell_26/TanhTanhGRU_1/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Tanh 
GRU_1/gru_cell_26/mul_1MulGRU_1/gru_cell_26/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_1w
GRU_1/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_26/sub/xЈ
GRU_1/gru_cell_26/subSub GRU_1/gru_cell_26/sub/x:output:0GRU_1/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/subЂ
GRU_1/gru_cell_26/mul_2MulGRU_1/gru_cell_26/sub:z:0GRU_1/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_2Ї
GRU_1/gru_cell_26/add_3AddV2GRU_1/gru_cell_26/mul_1:z:0GRU_1/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_3
#GRU_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_1/TensorArrayV2_1/element_shapeа
GRU_1/TensorArrayV2_1TensorListReserve,GRU_1/TensorArrayV2_1/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2_1Z

GRU_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_1/time
GRU_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_1/while/maximum_iterationsv
GRU_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_1/while/loop_counterў
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_26_readvariableop_resource0gru_1_gru_cell_26_matmul_readvariableop_resource2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_1_while_body_695157*#
condR
GRU_1_while_cond_695156*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_1/whileС
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_1/TensorArrayV2Stack/TensorListStackTensorListStackGRU_1/while:output:3?GRU_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_1/TensorArrayV2Stack/TensorListStack
GRU_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_1/strided_slice_3/stack
GRU_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_3/stack_1
GRU_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_3/stack_2О
GRU_1/strided_slice_3StridedSlice1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_1/strided_slice_3/stack:output:0&GRU_1/strided_slice_3/stack_1:output:0&GRU_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_3
GRU_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose_1/permН
GRU_1/transpose_1	Transpose1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0GRU_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transpose_1r
GRU_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/runtime_
GRU_2/ShapeShapeGRU_1/transpose_1:y:0*
T0*
_output_shapes
:2
GRU_2/Shape
GRU_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice/stack
GRU_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_1
GRU_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_2
GRU_2/strided_sliceStridedSliceGRU_2/Shape:output:0"GRU_2/strided_slice/stack:output:0$GRU_2/strided_slice/stack_1:output:0$GRU_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_sliceh
GRU_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/mul/y
GRU_2/zeros/mulMulGRU_2/strided_slice:output:0GRU_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/mulk
GRU_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_2/zeros/Less/y
GRU_2/zeros/LessLessGRU_2/zeros/mul:z:0GRU_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/Lessn
GRU_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/packed/1
GRU_2/zeros/packedPackGRU_2/strided_slice:output:0GRU_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_2/zeros/packedk
GRU_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/zeros/Const
GRU_2/zerosFillGRU_2/zeros/packed:output:0GRU_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/zeros
GRU_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose/perm
GRU_2/transpose	TransposeGRU_1/transpose_1:y:0GRU_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transposea
GRU_2/Shape_1ShapeGRU_2/transpose:y:0*
T0*
_output_shapes
:2
GRU_2/Shape_1
GRU_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_1/stack
GRU_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_1
GRU_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_2
GRU_2/strided_slice_1StridedSliceGRU_2/Shape_1:output:0$GRU_2/strided_slice_1/stack:output:0&GRU_2/strided_slice_1/stack_1:output:0&GRU_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_slice_1
!GRU_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/TensorArrayV2/element_shapeЪ
GRU_2/TensorArrayV2TensorListReserve*GRU_2/TensorArrayV2/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2Ы
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_2/transpose:y:0DGRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_2/TensorArrayUnstack/TensorListFromTensor
GRU_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_2/stack
GRU_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_1
GRU_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_2 
GRU_2/strided_slice_2StridedSliceGRU_2/transpose:y:0$GRU_2/strided_slice_2/stack:output:0&GRU_2/strided_slice_2/stack_1:output:0&GRU_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_2Ў
 GRU_2/gru_cell_27/ReadVariableOpReadVariableOp)gru_2_gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_27/ReadVariableOp 
GRU_2/gru_cell_27/unstackUnpack(GRU_2/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_27/unstackУ
'GRU_2/gru_cell_27/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_27/MatMul/ReadVariableOpС
GRU_2/gru_cell_27/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMulЛ
GRU_2/gru_cell_27/BiasAddBiasAdd"GRU_2/gru_cell_27/MatMul:product:0"GRU_2/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAddt
GRU_2/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_27/Const
!GRU_2/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_27/split/split_dimє
GRU_2/gru_cell_27/splitSplit*GRU_2/gru_cell_27/split/split_dim:output:0"GRU_2/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/splitЩ
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_27/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMul_1С
GRU_2/gru_cell_27/BiasAdd_1BiasAdd$GRU_2/gru_cell_27/MatMul_1:product:0"GRU_2/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAdd_1
GRU_2/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_27/Const_1
#GRU_2/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_27/split_1/split_dim­
GRU_2/gru_cell_27/split_1SplitV$GRU_2/gru_cell_27/BiasAdd_1:output:0"GRU_2/gru_cell_27/Const_1:output:0,GRU_2/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/split_1Џ
GRU_2/gru_cell_27/addAddV2 GRU_2/gru_cell_27/split:output:0"GRU_2/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add
GRU_2/gru_cell_27/SigmoidSigmoidGRU_2/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/SigmoidГ
GRU_2/gru_cell_27/add_1AddV2 GRU_2/gru_cell_27/split:output:1"GRU_2/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_1
GRU_2/gru_cell_27/Sigmoid_1SigmoidGRU_2/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Sigmoid_1Ќ
GRU_2/gru_cell_27/mulMulGRU_2/gru_cell_27/Sigmoid_1:y:0"GRU_2/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mulЊ
GRU_2/gru_cell_27/add_2AddV2 GRU_2/gru_cell_27/split:output:2GRU_2/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_2
GRU_2/gru_cell_27/TanhTanhGRU_2/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Tanh 
GRU_2/gru_cell_27/mul_1MulGRU_2/gru_cell_27/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_1w
GRU_2/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_27/sub/xЈ
GRU_2/gru_cell_27/subSub GRU_2/gru_cell_27/sub/x:output:0GRU_2/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/subЂ
GRU_2/gru_cell_27/mul_2MulGRU_2/gru_cell_27/sub:z:0GRU_2/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_2Ї
GRU_2/gru_cell_27/add_3AddV2GRU_2/gru_cell_27/mul_1:z:0GRU_2/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_3
#GRU_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_2/TensorArrayV2_1/element_shapeа
GRU_2/TensorArrayV2_1TensorListReserve,GRU_2/TensorArrayV2_1/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2_1Z

GRU_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_2/time
GRU_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_2/while/maximum_iterationsv
GRU_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_2/while/loop_counterў
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_27_readvariableop_resource0gru_2_gru_cell_27_matmul_readvariableop_resource2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_2_while_body_695312*#
condR
GRU_2_while_cond_695311*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_2/whileС
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_2/TensorArrayV2Stack/TensorListStackTensorListStackGRU_2/while:output:3?GRU_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_2/TensorArrayV2Stack/TensorListStack
GRU_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_2/strided_slice_3/stack
GRU_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_3/stack_1
GRU_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_3/stack_2О
GRU_2/strided_slice_3StridedSlice1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_2/strided_slice_3/stack:output:0&GRU_2/strided_slice_3/stack_1:output:0&GRU_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_3
GRU_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose_1/permН
GRU_2/transpose_1	Transpose1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0GRU_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transpose_1r
GRU_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/runtimeЂ
OUT/Tensordot/ReadVariableOpReadVariableOp%out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
OUT/Tensordot/ReadVariableOpr
OUT/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/axesy
OUT/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
OUT/Tensordot/freeo
OUT/Tensordot/ShapeShapeGRU_2/transpose_1:y:0*
T0*
_output_shapes
:2
OUT/Tensordot/Shape|
OUT/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2/axisх
OUT/Tensordot/GatherV2GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/free:output:0$OUT/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2
OUT/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2_1/axisы
OUT/Tensordot/GatherV2_1GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/axes:output:0&OUT/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2_1t
OUT/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const
OUT/Tensordot/ProdProdOUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prodx
OUT/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const_1
OUT/Tensordot/Prod_1Prod!OUT/Tensordot/GatherV2_1:output:0OUT/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prod_1x
OUT/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat/axisФ
OUT/Tensordot/concatConcatV2OUT/Tensordot/free:output:0OUT/Tensordot/axes:output:0"OUT/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat
OUT/Tensordot/stackPackOUT/Tensordot/Prod:output:0OUT/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/stackЋ
OUT/Tensordot/transpose	TransposeGRU_2/transpose_1:y:0OUT/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/transposeЏ
OUT/Tensordot/ReshapeReshapeOUT/Tensordot/transpose:y:0OUT/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
OUT/Tensordot/ReshapeЎ
OUT/Tensordot/MatMulMatMulOUT/Tensordot/Reshape:output:0$OUT/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/MatMulx
OUT/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/Const_2|
OUT/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat_1/axisб
OUT/Tensordot/concat_1ConcatV2OUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const_2:output:0$OUT/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat_1 
OUT/TensordotReshapeOUT/Tensordot/MatMul:product:0OUT/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot
OUT/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
OUT/BiasAdd/ReadVariableOp
OUT/BiasAddBiasAddOUT/Tensordot:output:0"OUT/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/BiasAddq
OUT/SigmoidSigmoidOUT/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Sigmoid
IdentityIdentityOUT/Sigmoid:y:0^GRU_1/while^GRU_2/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::2
GRU_1/whileGRU_1/while2
GRU_2/whileGRU_2/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
Њ
while_cond_693409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_693409___redundant_placeholder04
0while_while_cond_693409___redundant_placeholder14
0while_while_cond_693409___redundant_placeholder24
0while_while_cond_693409___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Б
к
+__inference_Supervisor_layer_call_fn_695812

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Supervisor_layer_call_and_return_conditional_losses_6943222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т
y
$__inference_OUT_layer_call_fn_697212

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_OUT_layer_call_and_return_conditional_losses_6942122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
нH
з
GRU_1_while_body_694774(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_26_readvariableop_resource_0<
8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_26_readvariableop_resource:
6gru_1_while_gru_cell_26_matmul_readvariableop_resource<
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resourceЯ
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFGRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_1/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_1/while/gru_cell_26/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_26/ReadVariableOpВ
GRU_1/while/gru_cell_26/unstackUnpack.GRU_1/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_26/unstackз
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_26/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_26/MatMulг
GRU_1/while/gru_cell_26/BiasAddBiasAdd(GRU_1/while/gru_cell_26/MatMul:product:0(GRU_1/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_26/BiasAdd
GRU_1/while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_26/Const
'GRU_1/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_26/split/split_dim
GRU_1/while/gru_cell_26/splitSplit0GRU_1/while/gru_cell_26/split/split_dim:output:0(GRU_1/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_26/splitн
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_26/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_26/MatMul_1й
!GRU_1/while/gru_cell_26/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_26/MatMul_1:product:0(GRU_1/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_26/BiasAdd_1
GRU_1/while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_26/Const_1Ё
)GRU_1/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_26/split_1/split_dimЫ
GRU_1/while/gru_cell_26/split_1SplitV*GRU_1/while/gru_cell_26/BiasAdd_1:output:0(GRU_1/while/gru_cell_26/Const_1:output:02GRU_1/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_26/split_1Ч
GRU_1/while/gru_cell_26/addAddV2&GRU_1/while/gru_cell_26/split:output:0(GRU_1/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add 
GRU_1/while/gru_cell_26/SigmoidSigmoidGRU_1/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_26/SigmoidЫ
GRU_1/while/gru_cell_26/add_1AddV2&GRU_1/while/gru_cell_26/split:output:1(GRU_1/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_1І
!GRU_1/while/gru_cell_26/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_26/Sigmoid_1Ф
GRU_1/while/gru_cell_26/mulMul%GRU_1/while/gru_cell_26/Sigmoid_1:y:0(GRU_1/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mulТ
GRU_1/while/gru_cell_26/add_2AddV2&GRU_1/while/gru_cell_26/split:output:2GRU_1/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_2
GRU_1/while/gru_cell_26/TanhTanh!GRU_1/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/TanhЗ
GRU_1/while/gru_cell_26/mul_1Mul#GRU_1/while/gru_cell_26/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_1
GRU_1/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_26/sub/xР
GRU_1/while/gru_cell_26/subSub&GRU_1/while/gru_cell_26/sub/x:output:0#GRU_1/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/subК
GRU_1/while/gru_cell_26/mul_2MulGRU_1/while/gru_cell_26/sub:z:0 GRU_1/while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_2П
GRU_1/while/gru_cell_26/add_3AddV2!GRU_1/while/gru_cell_26/mul_1:z:0!GRU_1/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_1/while/TensorArrayV2Write/TensorListSetItemh
GRU_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add/y
GRU_1/while/addAddV2gru_1_while_placeholderGRU_1/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/addl
GRU_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add_1/y
GRU_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_counterGRU_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/add_1p
GRU_1/while/IdentityIdentityGRU_1/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity
GRU_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_1/while/Identity_1r
GRU_1/while/Identity_2IdentityGRU_1/while/add:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_2
GRU_1/while/Identity_3Identity@GRU_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_3
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resource:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_26_matmul_readvariableop_resource8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_26_readvariableop_resource1gru_1_while_gru_cell_26_readvariableop_resource_0"5
gru_1_while_identityGRU_1/while/Identity:output:0"9
gru_1_while_identity_1GRU_1/while/Identity_1:output:0"9
gru_1_while_identity_2GRU_1/while/Identity_2:output:0"9
gru_1_while_identity_3GRU_1/while/Identity_3:output:0"9
gru_1_while_identity_4GRU_1/while/Identity_4:output:0"Р
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Н
І
__inference__traced_save_697475
file_prefix)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop7
3savev2_gru_1_gru_cell_26_kernel_read_readvariableopA
=savev2_gru_1_gru_cell_26_recurrent_kernel_read_readvariableop5
1savev2_gru_1_gru_cell_26_bias_read_readvariableop7
3savev2_gru_2_gru_cell_27_kernel_read_readvariableopA
=savev2_gru_2_gru_cell_27_recurrent_kernel_read_readvariableop5
1savev2_gru_2_gru_cell_27_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d375d5d5cf34480583f533f8f4d42328/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameџ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*
valueB	B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesм
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop3savev2_gru_1_gru_cell_26_kernel_read_readvariableop=savev2_gru_1_gru_cell_26_recurrent_kernel_read_readvariableop1savev2_gru_1_gru_cell_26_bias_read_readvariableop3savev2_gru_2_gru_cell_27_kernel_read_readvariableop=savev2_gru_2_gru_cell_27_recurrent_kernel_read_readvariableop1savev2_gru_2_gru_cell_27_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*c
_input_shapesR
P: :::<:<:<:<:<:<: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

:<:	

_output_shapes
: 


&__inference_GRU_1_layer_call_fn_696492

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_1_layer_call_and_return_conditional_losses_6938042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


&__inference_GRU_2_layer_call_fn_697161

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_2_layer_call_and_return_conditional_losses_6939922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю
ы
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_697360

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
о	
Ў
,__inference_gru_cell_26_layer_call_fn_697306

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_6924312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
а
Њ
while_cond_693713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_693713___redundant_placeholder04
0while_while_cond_693713___redundant_placeholder14
0while_while_cond_693713___redundant_placeholder24
0while_while_cond_693713___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ю
ы
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_697400

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ю!
л
while_body_693410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_27_693432_0
while_gru_cell_27_693434_0
while_gru_cell_27_693436_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_27_693432
while_gru_cell_27_693434
while_gru_cell_27_693436Ђ)while/gru_cell_27/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
)while/gru_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_27_693432_0while_gru_cell_27_693434_0while_gru_cell_27_693436_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_6930332+
)while/gru_cell_27/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_27/StatefulPartitionedCall:output:1*^while/gru_cell_27/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"6
while_gru_cell_27_693432while_gru_cell_27_693432_0"6
while_gru_cell_27_693434while_gru_cell_27_693434_0"6
while_gru_cell_27_693436while_gru_cell_27_693436_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_27/StatefulPartitionedCall)while/gru_cell_27/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
џ

GRU_1_while_cond_694432(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_694432___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_694432___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_694432___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_694432___redundant_placeholder3
gru_1_while_identity

GRU_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
GRU_1/while/Lesso
GRU_1/while/IdentityIdentityGRU_1/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_1/while/Identity"5
gru_1_while_identityGRU_1/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
@
Е
while_body_696901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_27_readvariableop_resource_06
2while_gru_cell_27_matmul_readvariableop_resource_08
4while_gru_cell_27_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_27_readvariableop_resource4
0while_gru_cell_27_matmul_readvariableop_resource6
2while_gru_cell_27_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_27/ReadVariableOp 
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_27/unstackХ
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_27/MatMul/ReadVariableOpг
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMulЛ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAddt
while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_27/Const
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_27/split/split_dimє
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/splitЫ
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_27/MatMul_1/ReadVariableOpМ
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMul_1С
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAdd_1
while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_27/Const_1
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_27/split_1/split_dim­
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0"while/gru_cell_27/Const_1:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/split_1Џ
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/SigmoidГ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_1
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Sigmoid_1Ќ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mulЊ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_2
while/gru_cell_27/TanhTanhwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Tanh
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_1w
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_27/sub/xЈ
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/subЂ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_2Ї
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Б

&__inference_GRU_2_layer_call_fn_696821
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_2_layer_call_and_return_conditional_losses_6933562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ю
ы
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_697252

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ос

F__inference_Supervisor_layer_call_and_return_conditional_losses_694705
gru_1_input-
)gru_1_gru_cell_26_readvariableop_resource4
0gru_1_gru_cell_26_matmul_readvariableop_resource6
2gru_1_gru_cell_26_matmul_1_readvariableop_resource-
)gru_2_gru_cell_27_readvariableop_resource4
0gru_2_gru_cell_27_matmul_readvariableop_resource6
2gru_2_gru_cell_27_matmul_1_readvariableop_resource)
%out_tensordot_readvariableop_resource'
#out_biasadd_readvariableop_resource
identityЂGRU_1/whileЂGRU_2/whileU
GRU_1/ShapeShapegru_1_input*
T0*
_output_shapes
:2
GRU_1/Shape
GRU_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice/stack
GRU_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_1
GRU_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_2
GRU_1/strided_sliceStridedSliceGRU_1/Shape:output:0"GRU_1/strided_slice/stack:output:0$GRU_1/strided_slice/stack_1:output:0$GRU_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_sliceh
GRU_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/mul/y
GRU_1/zeros/mulMulGRU_1/strided_slice:output:0GRU_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/mulk
GRU_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_1/zeros/Less/y
GRU_1/zeros/LessLessGRU_1/zeros/mul:z:0GRU_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/Lessn
GRU_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/packed/1
GRU_1/zeros/packedPackGRU_1/strided_slice:output:0GRU_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_1/zeros/packedk
GRU_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/zeros/Const
GRU_1/zerosFillGRU_1/zeros/packed:output:0GRU_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/zeros
GRU_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose/perm
GRU_1/transpose	Transposegru_1_inputGRU_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transposea
GRU_1/Shape_1ShapeGRU_1/transpose:y:0*
T0*
_output_shapes
:2
GRU_1/Shape_1
GRU_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_1/stack
GRU_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_1
GRU_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_2
GRU_1/strided_slice_1StridedSliceGRU_1/Shape_1:output:0$GRU_1/strided_slice_1/stack:output:0&GRU_1/strided_slice_1/stack_1:output:0&GRU_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_slice_1
!GRU_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/TensorArrayV2/element_shapeЪ
GRU_1/TensorArrayV2TensorListReserve*GRU_1/TensorArrayV2/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2Ы
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_1/transpose:y:0DGRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_1/TensorArrayUnstack/TensorListFromTensor
GRU_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_2/stack
GRU_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_1
GRU_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_2 
GRU_1/strided_slice_2StridedSliceGRU_1/transpose:y:0$GRU_1/strided_slice_2/stack:output:0&GRU_1/strided_slice_2/stack_1:output:0&GRU_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_2Ў
 GRU_1/gru_cell_26/ReadVariableOpReadVariableOp)gru_1_gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_26/ReadVariableOp 
GRU_1/gru_cell_26/unstackUnpack(GRU_1/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_26/unstackУ
'GRU_1/gru_cell_26/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_26/MatMul/ReadVariableOpС
GRU_1/gru_cell_26/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMulЛ
GRU_1/gru_cell_26/BiasAddBiasAdd"GRU_1/gru_cell_26/MatMul:product:0"GRU_1/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAddt
GRU_1/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_26/Const
!GRU_1/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_26/split/split_dimє
GRU_1/gru_cell_26/splitSplit*GRU_1/gru_cell_26/split/split_dim:output:0"GRU_1/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/splitЩ
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_26/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMul_1С
GRU_1/gru_cell_26/BiasAdd_1BiasAdd$GRU_1/gru_cell_26/MatMul_1:product:0"GRU_1/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAdd_1
GRU_1/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_26/Const_1
#GRU_1/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_26/split_1/split_dim­
GRU_1/gru_cell_26/split_1SplitV$GRU_1/gru_cell_26/BiasAdd_1:output:0"GRU_1/gru_cell_26/Const_1:output:0,GRU_1/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/split_1Џ
GRU_1/gru_cell_26/addAddV2 GRU_1/gru_cell_26/split:output:0"GRU_1/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add
GRU_1/gru_cell_26/SigmoidSigmoidGRU_1/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/SigmoidГ
GRU_1/gru_cell_26/add_1AddV2 GRU_1/gru_cell_26/split:output:1"GRU_1/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_1
GRU_1/gru_cell_26/Sigmoid_1SigmoidGRU_1/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Sigmoid_1Ќ
GRU_1/gru_cell_26/mulMulGRU_1/gru_cell_26/Sigmoid_1:y:0"GRU_1/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mulЊ
GRU_1/gru_cell_26/add_2AddV2 GRU_1/gru_cell_26/split:output:2GRU_1/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_2
GRU_1/gru_cell_26/TanhTanhGRU_1/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Tanh 
GRU_1/gru_cell_26/mul_1MulGRU_1/gru_cell_26/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_1w
GRU_1/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_26/sub/xЈ
GRU_1/gru_cell_26/subSub GRU_1/gru_cell_26/sub/x:output:0GRU_1/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/subЂ
GRU_1/gru_cell_26/mul_2MulGRU_1/gru_cell_26/sub:z:0GRU_1/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_2Ї
GRU_1/gru_cell_26/add_3AddV2GRU_1/gru_cell_26/mul_1:z:0GRU_1/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_3
#GRU_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_1/TensorArrayV2_1/element_shapeа
GRU_1/TensorArrayV2_1TensorListReserve,GRU_1/TensorArrayV2_1/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2_1Z

GRU_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_1/time
GRU_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_1/while/maximum_iterationsv
GRU_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_1/while/loop_counterў
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_26_readvariableop_resource0gru_1_gru_cell_26_matmul_readvariableop_resource2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_1_while_body_694433*#
condR
GRU_1_while_cond_694432*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_1/whileС
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_1/TensorArrayV2Stack/TensorListStackTensorListStackGRU_1/while:output:3?GRU_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_1/TensorArrayV2Stack/TensorListStack
GRU_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_1/strided_slice_3/stack
GRU_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_3/stack_1
GRU_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_3/stack_2О
GRU_1/strided_slice_3StridedSlice1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_1/strided_slice_3/stack:output:0&GRU_1/strided_slice_3/stack_1:output:0&GRU_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_3
GRU_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose_1/permН
GRU_1/transpose_1	Transpose1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0GRU_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transpose_1r
GRU_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/runtime_
GRU_2/ShapeShapeGRU_1/transpose_1:y:0*
T0*
_output_shapes
:2
GRU_2/Shape
GRU_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice/stack
GRU_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_1
GRU_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_2
GRU_2/strided_sliceStridedSliceGRU_2/Shape:output:0"GRU_2/strided_slice/stack:output:0$GRU_2/strided_slice/stack_1:output:0$GRU_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_sliceh
GRU_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/mul/y
GRU_2/zeros/mulMulGRU_2/strided_slice:output:0GRU_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/mulk
GRU_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_2/zeros/Less/y
GRU_2/zeros/LessLessGRU_2/zeros/mul:z:0GRU_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/Lessn
GRU_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/packed/1
GRU_2/zeros/packedPackGRU_2/strided_slice:output:0GRU_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_2/zeros/packedk
GRU_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/zeros/Const
GRU_2/zerosFillGRU_2/zeros/packed:output:0GRU_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/zeros
GRU_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose/perm
GRU_2/transpose	TransposeGRU_1/transpose_1:y:0GRU_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transposea
GRU_2/Shape_1ShapeGRU_2/transpose:y:0*
T0*
_output_shapes
:2
GRU_2/Shape_1
GRU_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_1/stack
GRU_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_1
GRU_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_2
GRU_2/strided_slice_1StridedSliceGRU_2/Shape_1:output:0$GRU_2/strided_slice_1/stack:output:0&GRU_2/strided_slice_1/stack_1:output:0&GRU_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_slice_1
!GRU_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/TensorArrayV2/element_shapeЪ
GRU_2/TensorArrayV2TensorListReserve*GRU_2/TensorArrayV2/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2Ы
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_2/transpose:y:0DGRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_2/TensorArrayUnstack/TensorListFromTensor
GRU_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_2/stack
GRU_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_1
GRU_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_2 
GRU_2/strided_slice_2StridedSliceGRU_2/transpose:y:0$GRU_2/strided_slice_2/stack:output:0&GRU_2/strided_slice_2/stack_1:output:0&GRU_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_2Ў
 GRU_2/gru_cell_27/ReadVariableOpReadVariableOp)gru_2_gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_27/ReadVariableOp 
GRU_2/gru_cell_27/unstackUnpack(GRU_2/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_27/unstackУ
'GRU_2/gru_cell_27/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_27/MatMul/ReadVariableOpС
GRU_2/gru_cell_27/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMulЛ
GRU_2/gru_cell_27/BiasAddBiasAdd"GRU_2/gru_cell_27/MatMul:product:0"GRU_2/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAddt
GRU_2/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_27/Const
!GRU_2/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_27/split/split_dimє
GRU_2/gru_cell_27/splitSplit*GRU_2/gru_cell_27/split/split_dim:output:0"GRU_2/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/splitЩ
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_27/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMul_1С
GRU_2/gru_cell_27/BiasAdd_1BiasAdd$GRU_2/gru_cell_27/MatMul_1:product:0"GRU_2/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAdd_1
GRU_2/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_27/Const_1
#GRU_2/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_27/split_1/split_dim­
GRU_2/gru_cell_27/split_1SplitV$GRU_2/gru_cell_27/BiasAdd_1:output:0"GRU_2/gru_cell_27/Const_1:output:0,GRU_2/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/split_1Џ
GRU_2/gru_cell_27/addAddV2 GRU_2/gru_cell_27/split:output:0"GRU_2/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add
GRU_2/gru_cell_27/SigmoidSigmoidGRU_2/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/SigmoidГ
GRU_2/gru_cell_27/add_1AddV2 GRU_2/gru_cell_27/split:output:1"GRU_2/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_1
GRU_2/gru_cell_27/Sigmoid_1SigmoidGRU_2/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Sigmoid_1Ќ
GRU_2/gru_cell_27/mulMulGRU_2/gru_cell_27/Sigmoid_1:y:0"GRU_2/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mulЊ
GRU_2/gru_cell_27/add_2AddV2 GRU_2/gru_cell_27/split:output:2GRU_2/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_2
GRU_2/gru_cell_27/TanhTanhGRU_2/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Tanh 
GRU_2/gru_cell_27/mul_1MulGRU_2/gru_cell_27/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_1w
GRU_2/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_27/sub/xЈ
GRU_2/gru_cell_27/subSub GRU_2/gru_cell_27/sub/x:output:0GRU_2/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/subЂ
GRU_2/gru_cell_27/mul_2MulGRU_2/gru_cell_27/sub:z:0GRU_2/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_2Ї
GRU_2/gru_cell_27/add_3AddV2GRU_2/gru_cell_27/mul_1:z:0GRU_2/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_3
#GRU_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_2/TensorArrayV2_1/element_shapeа
GRU_2/TensorArrayV2_1TensorListReserve,GRU_2/TensorArrayV2_1/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2_1Z

GRU_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_2/time
GRU_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_2/while/maximum_iterationsv
GRU_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_2/while/loop_counterў
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_27_readvariableop_resource0gru_2_gru_cell_27_matmul_readvariableop_resource2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_2_while_body_694588*#
condR
GRU_2_while_cond_694587*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_2/whileС
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_2/TensorArrayV2Stack/TensorListStackTensorListStackGRU_2/while:output:3?GRU_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_2/TensorArrayV2Stack/TensorListStack
GRU_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_2/strided_slice_3/stack
GRU_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_3/stack_1
GRU_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_3/stack_2О
GRU_2/strided_slice_3StridedSlice1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_2/strided_slice_3/stack:output:0&GRU_2/strided_slice_3/stack_1:output:0&GRU_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_3
GRU_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose_1/permН
GRU_2/transpose_1	Transpose1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0GRU_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transpose_1r
GRU_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/runtimeЂ
OUT/Tensordot/ReadVariableOpReadVariableOp%out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
OUT/Tensordot/ReadVariableOpr
OUT/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/axesy
OUT/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
OUT/Tensordot/freeo
OUT/Tensordot/ShapeShapeGRU_2/transpose_1:y:0*
T0*
_output_shapes
:2
OUT/Tensordot/Shape|
OUT/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2/axisх
OUT/Tensordot/GatherV2GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/free:output:0$OUT/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2
OUT/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2_1/axisы
OUT/Tensordot/GatherV2_1GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/axes:output:0&OUT/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2_1t
OUT/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const
OUT/Tensordot/ProdProdOUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prodx
OUT/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const_1
OUT/Tensordot/Prod_1Prod!OUT/Tensordot/GatherV2_1:output:0OUT/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prod_1x
OUT/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat/axisФ
OUT/Tensordot/concatConcatV2OUT/Tensordot/free:output:0OUT/Tensordot/axes:output:0"OUT/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat
OUT/Tensordot/stackPackOUT/Tensordot/Prod:output:0OUT/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/stackЋ
OUT/Tensordot/transpose	TransposeGRU_2/transpose_1:y:0OUT/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/transposeЏ
OUT/Tensordot/ReshapeReshapeOUT/Tensordot/transpose:y:0OUT/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
OUT/Tensordot/ReshapeЎ
OUT/Tensordot/MatMulMatMulOUT/Tensordot/Reshape:output:0$OUT/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/MatMulx
OUT/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/Const_2|
OUT/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat_1/axisб
OUT/Tensordot/concat_1ConcatV2OUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const_2:output:0$OUT/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat_1 
OUT/TensordotReshapeOUT/Tensordot/MatMul:product:0OUT/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot
OUT/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
OUT/BiasAdd/ReadVariableOp
OUT/BiasAddBiasAddOUT/Tensordot:output:0"OUT/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/BiasAddq
OUT/SigmoidSigmoidOUT/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Sigmoid
IdentityIdentityOUT/Sigmoid:y:0^GRU_1/while^GRU_2/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::2
GRU_1/whileGRU_1/while2
GRU_2/whileGRU_2/while:X T
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameGRU_1_input
@
Е
while_body_693555
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_26_readvariableop_resource_06
2while_gru_cell_26_matmul_readvariableop_resource_08
4while_gru_cell_26_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_26_readvariableop_resource4
0while_gru_cell_26_matmul_readvariableop_resource6
2while_gru_cell_26_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_26/ReadVariableOp 
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_26/unstackХ
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_26/MatMul/ReadVariableOpг
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMulЛ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAddt
while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_26/Const
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_26/split/split_dimє
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/splitЫ
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_26/MatMul_1/ReadVariableOpМ
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMul_1С
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAdd_1
while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_26/Const_1
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_26/split_1/split_dim­
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0"while/gru_cell_26/Const_1:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/split_1Џ
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/SigmoidГ
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_1
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Sigmoid_1Ќ
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mulЊ
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_2
while/gru_cell_26/TanhTanhwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Tanh
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_1w
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_26/sub/xЈ
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/subЂ
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_2Ї
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
@
Е
while_body_696380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_26_readvariableop_resource_06
2while_gru_cell_26_matmul_readvariableop_resource_08
4while_gru_cell_26_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_26_readvariableop_resource4
0while_gru_cell_26_matmul_readvariableop_resource6
2while_gru_cell_26_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_26/ReadVariableOp 
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_26/unstackХ
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_26/MatMul/ReadVariableOpг
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMulЛ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAddt
while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_26/Const
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_26/split/split_dimє
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/splitЫ
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_26/MatMul_1/ReadVariableOpМ
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMul_1С
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAdd_1
while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_26/Const_1
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_26/split_1/split_dim­
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0"while/gru_cell_26/Const_1:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/split_1Џ
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/SigmoidГ
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_1
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Sigmoid_1Ќ
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mulЊ
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_2
while/gru_cell_26/TanhTanhwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Tanh
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_1w
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_26/sub/xЈ
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/subЂ
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_2Ї
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
о	
Ў
,__inference_gru_cell_26_layer_call_fn_697320

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_6924712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
а
Њ
while_cond_694060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_694060___redundant_placeholder04
0while_while_cond_694060___redundant_placeholder14
0while_while_cond_694060___redundant_placeholder24
0while_while_cond_694060___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Р
п
+__inference_Supervisor_layer_call_fn_695088
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Supervisor_layer_call_and_return_conditional_losses_6943222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameGRU_1_input
ю!
л
while_body_692730
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_26_692752_0
while_gru_cell_26_692754_0
while_gru_cell_26_692756_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_26_692752
while_gru_cell_26_692754
while_gru_cell_26_692756Ђ)while/gru_cell_26/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
)while/gru_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_26_692752_0while_gru_cell_26_692754_0while_gru_cell_26_692756_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_6924312+
)while/gru_cell_26/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_26/StatefulPartitionedCall:output:1*^while/gru_cell_26/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"6
while_gru_cell_26_692752while_gru_cell_26_692752_0"6
while_gru_cell_26_692754while_gru_cell_26_692754_0"6
while_gru_cell_26_692756while_gru_cell_26_692756_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_26/StatefulPartitionedCall)while/gru_cell_26/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
@
Е
while_body_696221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_26_readvariableop_resource_06
2while_gru_cell_26_matmul_readvariableop_resource_08
4while_gru_cell_26_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_26_readvariableop_resource4
0while_gru_cell_26_matmul_readvariableop_resource6
2while_gru_cell_26_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_26/ReadVariableOp 
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_26/unstackХ
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_26/MatMul/ReadVariableOpг
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMulЛ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAddt
while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_26/Const
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_26/split/split_dimє
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/splitЫ
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_26/MatMul_1/ReadVariableOpМ
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMul_1С
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAdd_1
while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_26/Const_1
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_26/split_1/split_dim­
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0"while/gru_cell_26/Const_1:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/split_1Џ
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/SigmoidГ
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_1
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Sigmoid_1Ќ
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mulЊ
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_2
while/gru_cell_26/TanhTanhwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Tanh
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_1w
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_26/sub/xЈ
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/subЂ
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_2Ї
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЇX
ѕ
A__inference_GRU_1_layer_call_and_return_conditional_losses_696130
inputs_0'
#gru_cell_26_readvariableop_resource.
*gru_cell_26_matmul_readvariableop_resource0
,gru_cell_26_matmul_1_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_26/ReadVariableOp
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_26/unstackБ
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_26/MatMul/ReadVariableOpЉ
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMulЃ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAddh
gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_26/Const
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split/split_dimм
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/splitЗ
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_26/MatMul_1/ReadVariableOpЅ
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMul_1Љ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAdd_1
gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_26/Const_1
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split_1/split_dim
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const_1:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/split_1
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add|
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_1
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid_1
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_2u
gru_cell_26/TanhTanhgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Tanh
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_1k
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_26/sub/x
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/sub
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_2
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_696040*
condR
while_cond_696039*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
џ

GRU_2_while_cond_694587(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1@
<gru_2_while_gru_2_while_cond_694587___redundant_placeholder0@
<gru_2_while_gru_2_while_cond_694587___redundant_placeholder1@
<gru_2_while_gru_2_while_cond_694587___redundant_placeholder2@
<gru_2_while_gru_2_while_cond_694587___redundant_placeholder3
gru_2_while_identity

GRU_2/while/LessLessgru_2_while_placeholder&gru_2_while_less_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
GRU_2/while/Lesso
GRU_2/while/IdentityIdentityGRU_2/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_2/while/Identity"5
gru_2_while_identityGRU_2/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ЉX


"Supervisor_GRU_2_while_body_692242>
:supervisor_gru_2_while_supervisor_gru_2_while_loop_counterD
@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations&
"supervisor_gru_2_while_placeholder(
$supervisor_gru_2_while_placeholder_1(
$supervisor_gru_2_while_placeholder_2=
9supervisor_gru_2_while_supervisor_gru_2_strided_slice_1_0y
usupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor_0@
<supervisor_gru_2_while_gru_cell_27_readvariableop_resource_0G
Csupervisor_gru_2_while_gru_cell_27_matmul_readvariableop_resource_0I
Esupervisor_gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0#
supervisor_gru_2_while_identity%
!supervisor_gru_2_while_identity_1%
!supervisor_gru_2_while_identity_2%
!supervisor_gru_2_while_identity_3%
!supervisor_gru_2_while_identity_4;
7supervisor_gru_2_while_supervisor_gru_2_strided_slice_1w
ssupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor>
:supervisor_gru_2_while_gru_cell_27_readvariableop_resourceE
Asupervisor_gru_2_while_gru_cell_27_matmul_readvariableop_resourceG
Csupervisor_gru_2_while_gru_cell_27_matmul_1_readvariableop_resourceх
HSupervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
HSupervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:Supervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor_0"supervisor_gru_2_while_placeholderQSupervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:Supervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItemу
1Supervisor/GRU_2/while/gru_cell_27/ReadVariableOpReadVariableOp<supervisor_gru_2_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype023
1Supervisor/GRU_2/while/gru_cell_27/ReadVariableOpг
*Supervisor/GRU_2/while/gru_cell_27/unstackUnpack9Supervisor/GRU_2/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2,
*Supervisor/GRU_2/while/gru_cell_27/unstackј
8Supervisor/GRU_2/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOpCsupervisor_gru_2_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02:
8Supervisor/GRU_2/while/gru_cell_27/MatMul/ReadVariableOp
)Supervisor/GRU_2/while/gru_cell_27/MatMulMatMulASupervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:0@Supervisor/GRU_2/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2+
)Supervisor/GRU_2/while/gru_cell_27/MatMulџ
*Supervisor/GRU_2/while/gru_cell_27/BiasAddBiasAdd3Supervisor/GRU_2/while/gru_cell_27/MatMul:product:03Supervisor/GRU_2/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2,
*Supervisor/GRU_2/while/gru_cell_27/BiasAdd
(Supervisor/GRU_2/while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(Supervisor/GRU_2/while/gru_cell_27/ConstГ
2Supervisor/GRU_2/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2Supervisor/GRU_2/while/gru_cell_27/split/split_dimИ
(Supervisor/GRU_2/while/gru_cell_27/splitSplit;Supervisor/GRU_2/while/gru_cell_27/split/split_dim:output:03Supervisor/GRU_2/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2*
(Supervisor/GRU_2/while/gru_cell_27/splitў
:Supervisor/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOpEsupervisor_gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02<
:Supervisor/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOp
+Supervisor/GRU_2/while/gru_cell_27/MatMul_1MatMul$supervisor_gru_2_while_placeholder_2BSupervisor/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2-
+Supervisor/GRU_2/while/gru_cell_27/MatMul_1
,Supervisor/GRU_2/while/gru_cell_27/BiasAdd_1BiasAdd5Supervisor/GRU_2/while/gru_cell_27/MatMul_1:product:03Supervisor/GRU_2/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2.
,Supervisor/GRU_2/while/gru_cell_27/BiasAdd_1­
*Supervisor/GRU_2/while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2,
*Supervisor/GRU_2/while/gru_cell_27/Const_1З
4Supervisor/GRU_2/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ26
4Supervisor/GRU_2/while/gru_cell_27/split_1/split_dim
*Supervisor/GRU_2/while/gru_cell_27/split_1SplitV5Supervisor/GRU_2/while/gru_cell_27/BiasAdd_1:output:03Supervisor/GRU_2/while/gru_cell_27/Const_1:output:0=Supervisor/GRU_2/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*Supervisor/GRU_2/while/gru_cell_27/split_1ѓ
&Supervisor/GRU_2/while/gru_cell_27/addAddV21Supervisor/GRU_2/while/gru_cell_27/split:output:03Supervisor/GRU_2/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_27/addС
*Supervisor/GRU_2/while/gru_cell_27/SigmoidSigmoid*Supervisor/GRU_2/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*Supervisor/GRU_2/while/gru_cell_27/Sigmoidї
(Supervisor/GRU_2/while/gru_cell_27/add_1AddV21Supervisor/GRU_2/while/gru_cell_27/split:output:13Supervisor/GRU_2/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_27/add_1Ч
,Supervisor/GRU_2/while/gru_cell_27/Sigmoid_1Sigmoid,Supervisor/GRU_2/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,Supervisor/GRU_2/while/gru_cell_27/Sigmoid_1№
&Supervisor/GRU_2/while/gru_cell_27/mulMul0Supervisor/GRU_2/while/gru_cell_27/Sigmoid_1:y:03Supervisor/GRU_2/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_27/mulю
(Supervisor/GRU_2/while/gru_cell_27/add_2AddV21Supervisor/GRU_2/while/gru_cell_27/split:output:2*Supervisor/GRU_2/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_27/add_2К
'Supervisor/GRU_2/while/gru_cell_27/TanhTanh,Supervisor/GRU_2/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'Supervisor/GRU_2/while/gru_cell_27/Tanhу
(Supervisor/GRU_2/while/gru_cell_27/mul_1Mul.Supervisor/GRU_2/while/gru_cell_27/Sigmoid:y:0$supervisor_gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_27/mul_1
(Supervisor/GRU_2/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(Supervisor/GRU_2/while/gru_cell_27/sub/xь
&Supervisor/GRU_2/while/gru_cell_27/subSub1Supervisor/GRU_2/while/gru_cell_27/sub/x:output:0.Supervisor/GRU_2/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_27/subц
(Supervisor/GRU_2/while/gru_cell_27/mul_2Mul*Supervisor/GRU_2/while/gru_cell_27/sub:z:0+Supervisor/GRU_2/while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_27/mul_2ы
(Supervisor/GRU_2/while/gru_cell_27/add_3AddV2,Supervisor/GRU_2/while/gru_cell_27/mul_1:z:0,Supervisor/GRU_2/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_27/add_3Д
;Supervisor/GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$supervisor_gru_2_while_placeholder_1"supervisor_gru_2_while_placeholder,Supervisor/GRU_2/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02=
;Supervisor/GRU_2/while/TensorArrayV2Write/TensorListSetItem~
Supervisor/GRU_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
Supervisor/GRU_2/while/add/y­
Supervisor/GRU_2/while/addAddV2"supervisor_gru_2_while_placeholder%Supervisor/GRU_2/while/add/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_2/while/add
Supervisor/GRU_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
Supervisor/GRU_2/while/add_1/yЫ
Supervisor/GRU_2/while/add_1AddV2:supervisor_gru_2_while_supervisor_gru_2_while_loop_counter'Supervisor/GRU_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_2/while/add_1
Supervisor/GRU_2/while/IdentityIdentity Supervisor/GRU_2/while/add_1:z:0*
T0*
_output_shapes
: 2!
Supervisor/GRU_2/while/IdentityЕ
!Supervisor/GRU_2/while/Identity_1Identity@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations*
T0*
_output_shapes
: 2#
!Supervisor/GRU_2/while/Identity_1
!Supervisor/GRU_2/while/Identity_2IdentitySupervisor/GRU_2/while/add:z:0*
T0*
_output_shapes
: 2#
!Supervisor/GRU_2/while/Identity_2Р
!Supervisor/GRU_2/while/Identity_3IdentityKSupervisor/GRU_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2#
!Supervisor/GRU_2/while/Identity_3В
!Supervisor/GRU_2/while/Identity_4Identity,Supervisor/GRU_2/while/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_2/while/Identity_4"
Csupervisor_gru_2_while_gru_cell_27_matmul_1_readvariableop_resourceEsupervisor_gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0"
Asupervisor_gru_2_while_gru_cell_27_matmul_readvariableop_resourceCsupervisor_gru_2_while_gru_cell_27_matmul_readvariableop_resource_0"z
:supervisor_gru_2_while_gru_cell_27_readvariableop_resource<supervisor_gru_2_while_gru_cell_27_readvariableop_resource_0"K
supervisor_gru_2_while_identity(Supervisor/GRU_2/while/Identity:output:0"O
!supervisor_gru_2_while_identity_1*Supervisor/GRU_2/while/Identity_1:output:0"O
!supervisor_gru_2_while_identity_2*Supervisor/GRU_2/while/Identity_2:output:0"O
!supervisor_gru_2_while_identity_3*Supervisor/GRU_2/while/Identity_3:output:0"O
!supervisor_gru_2_while_identity_4*Supervisor/GRU_2/while/Identity_4:output:0"t
7supervisor_gru_2_while_supervisor_gru_2_strided_slice_19supervisor_gru_2_while_supervisor_gru_2_strided_slice_1_0"ь
ssupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensorusupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
щ<
ж
A__inference_GRU_1_layer_call_and_return_conditional_losses_692912

inputs
gru_cell_26_692836
gru_cell_26_692838
gru_cell_26_692840
identityЂ#gru_cell_26/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2№
#gru_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_26_692836gru_cell_26_692838gru_cell_26_692840*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_6924712%
#gru_cell_26/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_26_692836gru_cell_26_692838gru_cell_26_692840*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_692848*
condR
while_cond_692847*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0$^gru_cell_26/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_26/StatefulPartitionedCall#gru_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Б

&__inference_GRU_2_layer_call_fn_696832
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_2_layer_call_and_return_conditional_losses_6934742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
шW
ѓ
A__inference_GRU_1_layer_call_and_return_conditional_losses_693645

inputs'
#gru_cell_26_readvariableop_resource.
*gru_cell_26_matmul_readvariableop_resource0
,gru_cell_26_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_26/ReadVariableOp
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_26/unstackБ
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_26/MatMul/ReadVariableOpЉ
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMulЃ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAddh
gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_26/Const
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split/split_dimм
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/splitЗ
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_26/MatMul_1/ReadVariableOpЅ
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMul_1Љ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAdd_1
gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_26/Const_1
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split_1/split_dim
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const_1:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/split_1
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add|
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_1
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid_1
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_2u
gru_cell_26/TanhTanhgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Tanh
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_1k
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_26/sub/x
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/sub
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_2
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_693555*
condR
while_cond_693554*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б
к
+__inference_Supervisor_layer_call_fn_695791

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Supervisor_layer_call_and_return_conditional_losses_6942782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
нH
з
GRU_1_while_body_694433(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_26_readvariableop_resource_0<
8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_26_readvariableop_resource:
6gru_1_while_gru_cell_26_matmul_readvariableop_resource<
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resourceЯ
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFGRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_1/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_1/while/gru_cell_26/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_26/ReadVariableOpВ
GRU_1/while/gru_cell_26/unstackUnpack.GRU_1/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_26/unstackз
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_26/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_26/MatMulг
GRU_1/while/gru_cell_26/BiasAddBiasAdd(GRU_1/while/gru_cell_26/MatMul:product:0(GRU_1/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_26/BiasAdd
GRU_1/while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_26/Const
'GRU_1/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_26/split/split_dim
GRU_1/while/gru_cell_26/splitSplit0GRU_1/while/gru_cell_26/split/split_dim:output:0(GRU_1/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_26/splitн
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_26/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_26/MatMul_1й
!GRU_1/while/gru_cell_26/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_26/MatMul_1:product:0(GRU_1/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_26/BiasAdd_1
GRU_1/while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_26/Const_1Ё
)GRU_1/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_26/split_1/split_dimЫ
GRU_1/while/gru_cell_26/split_1SplitV*GRU_1/while/gru_cell_26/BiasAdd_1:output:0(GRU_1/while/gru_cell_26/Const_1:output:02GRU_1/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_26/split_1Ч
GRU_1/while/gru_cell_26/addAddV2&GRU_1/while/gru_cell_26/split:output:0(GRU_1/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add 
GRU_1/while/gru_cell_26/SigmoidSigmoidGRU_1/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_26/SigmoidЫ
GRU_1/while/gru_cell_26/add_1AddV2&GRU_1/while/gru_cell_26/split:output:1(GRU_1/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_1І
!GRU_1/while/gru_cell_26/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_26/Sigmoid_1Ф
GRU_1/while/gru_cell_26/mulMul%GRU_1/while/gru_cell_26/Sigmoid_1:y:0(GRU_1/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mulТ
GRU_1/while/gru_cell_26/add_2AddV2&GRU_1/while/gru_cell_26/split:output:2GRU_1/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_2
GRU_1/while/gru_cell_26/TanhTanh!GRU_1/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/TanhЗ
GRU_1/while/gru_cell_26/mul_1Mul#GRU_1/while/gru_cell_26/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_1
GRU_1/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_26/sub/xР
GRU_1/while/gru_cell_26/subSub&GRU_1/while/gru_cell_26/sub/x:output:0#GRU_1/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/subК
GRU_1/while/gru_cell_26/mul_2MulGRU_1/while/gru_cell_26/sub:z:0 GRU_1/while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_2П
GRU_1/while/gru_cell_26/add_3AddV2!GRU_1/while/gru_cell_26/mul_1:z:0!GRU_1/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_1/while/TensorArrayV2Write/TensorListSetItemh
GRU_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add/y
GRU_1/while/addAddV2gru_1_while_placeholderGRU_1/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/addl
GRU_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add_1/y
GRU_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_counterGRU_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/add_1p
GRU_1/while/IdentityIdentityGRU_1/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity
GRU_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_1/while/Identity_1r
GRU_1/while/Identity_2IdentityGRU_1/while/add:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_2
GRU_1/while/Identity_3Identity@GRU_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_3
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resource:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_26_matmul_readvariableop_resource8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_26_readvariableop_resource1gru_1_while_gru_cell_26_readvariableop_resource_0"5
gru_1_while_identityGRU_1/while/Identity:output:0"9
gru_1_while_identity_1GRU_1/while/Identity_1:output:0"9
gru_1_while_identity_2GRU_1/while/Identity_2:output:0"9
gru_1_while_identity_3GRU_1/while/Identity_3:output:0"9
gru_1_while_identity_4GRU_1/while/Identity_4:output:0"Р
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
а
Њ
while_cond_697059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_697059___redundant_placeholder04
0while_while_cond_697059___redundant_placeholder14
0while_while_cond_697059___redundant_placeholder24
0while_while_cond_697059___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
о	
Ў
,__inference_gru_cell_27_layer_call_fn_697414

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_6929932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ц
щ
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_692431

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
а
Њ
while_cond_693554
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_693554___redundant_placeholder04
0while_while_cond_693554___redundant_placeholder14
0while_while_cond_693554___redundant_placeholder24
0while_while_cond_693554___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
шW
ѓ
A__inference_GRU_1_layer_call_and_return_conditional_losses_696311

inputs'
#gru_cell_26_readvariableop_resource.
*gru_cell_26_matmul_readvariableop_resource0
,gru_cell_26_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_26/ReadVariableOp
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_26/unstackБ
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_26/MatMul/ReadVariableOpЉ
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMulЃ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAddh
gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_26/Const
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split/split_dimм
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/splitЗ
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_26/MatMul_1/ReadVariableOpЅ
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMul_1Љ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAdd_1
gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_26/Const_1
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split_1/split_dim
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const_1:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/split_1
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add|
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_1
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid_1
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_2u
gru_cell_26/TanhTanhgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Tanh
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_1k
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_26/sub/x
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/sub
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_2
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_696221*
condR
while_cond_696220*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н
Њ
?__inference_OUT_layer_call_and_return_conditional_losses_694212

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
BiasAdde
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
Sigmoidc
IdentityIdentitySigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:::S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

GRU_2_while_cond_694928(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1@
<gru_2_while_gru_2_while_cond_694928___redundant_placeholder0@
<gru_2_while_gru_2_while_cond_694928___redundant_placeholder1@
<gru_2_while_gru_2_while_cond_694928___redundant_placeholder2@
<gru_2_while_gru_2_while_cond_694928___redundant_placeholder3
gru_2_while_identity

GRU_2/while/LessLessgru_2_while_placeholder&gru_2_while_less_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
GRU_2/while/Lesso
GRU_2/while/IdentityIdentityGRU_2/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_2/while/Identity"5
gru_2_while_identityGRU_2/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
шW
ѓ
A__inference_GRU_2_layer_call_and_return_conditional_losses_697150

inputs'
#gru_cell_27_readvariableop_resource.
*gru_cell_27_matmul_readvariableop_resource0
,gru_cell_27_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_27/ReadVariableOp
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_27/unstackБ
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_27/MatMul/ReadVariableOpЉ
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMulЃ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAddh
gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_27/Const
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split/split_dimм
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/splitЗ
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_27/MatMul_1/ReadVariableOpЅ
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMul_1Љ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAdd_1
gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_27/Const_1
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split_1/split_dim
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const_1:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/split_1
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add|
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_1
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid_1
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_2u
gru_cell_27/TanhTanhgru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Tanh
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_1k
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_27/sub/x
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/sub
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_2
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_697060*
condR
while_cond_697059*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
@
Е
while_body_694061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_27_readvariableop_resource_06
2while_gru_cell_27_matmul_readvariableop_resource_08
4while_gru_cell_27_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_27_readvariableop_resource4
0while_gru_cell_27_matmul_readvariableop_resource6
2while_gru_cell_27_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_27/ReadVariableOp 
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_27/unstackХ
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_27/MatMul/ReadVariableOpг
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMulЛ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAddt
while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_27/Const
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_27/split/split_dimє
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/splitЫ
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_27/MatMul_1/ReadVariableOpМ
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMul_1С
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAdd_1
while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_27/Const_1
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_27/split_1/split_dim­
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0"while/gru_cell_27/Const_1:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/split_1Џ
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/SigmoidГ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_1
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Sigmoid_1Ќ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mulЊ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_2
while/gru_cell_27/TanhTanhwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Tanh
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_1w
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_27/sub/xЈ
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/subЂ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_2Ї
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
шW
ѓ
A__inference_GRU_1_layer_call_and_return_conditional_losses_693804

inputs'
#gru_cell_26_readvariableop_resource.
*gru_cell_26_matmul_readvariableop_resource0
,gru_cell_26_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_26/ReadVariableOp
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_26/unstackБ
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_26/MatMul/ReadVariableOpЉ
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMulЃ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAddh
gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_26/Const
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split/split_dimм
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/splitЗ
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_26/MatMul_1/ReadVariableOpЅ
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMul_1Љ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAdd_1
gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_26/Const_1
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split_1/split_dim
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const_1:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/split_1
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add|
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_1
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid_1
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_2u
gru_cell_26/TanhTanhgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Tanh
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_1k
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_26/sub/x
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/sub
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_2
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_693714*
condR
while_cond_693713*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
Њ
while_cond_696220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_696220___redundant_placeholder04
0while_while_cond_696220___redundant_placeholder14
0while_while_cond_696220___redundant_placeholder24
0while_while_cond_696220___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
&
ц
"__inference__traced_restore_697509
file_prefix
assignvariableop_out_kernel
assignvariableop_1_out_bias/
+assignvariableop_2_gru_1_gru_cell_26_kernel9
5assignvariableop_3_gru_1_gru_cell_26_recurrent_kernel-
)assignvariableop_4_gru_1_gru_cell_26_bias/
+assignvariableop_5_gru_2_gru_cell_27_kernel9
5assignvariableop_6_gru_2_gru_cell_27_recurrent_kernel-
)assignvariableop_7_gru_2_gru_cell_27_bias

identity_9ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*
valueB	B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesи
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_out_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1 
AssignVariableOp_1AssignVariableOpassignvariableop_1_out_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2А
AssignVariableOp_2AssignVariableOp+assignvariableop_2_gru_1_gru_cell_26_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3К
AssignVariableOp_3AssignVariableOp5assignvariableop_3_gru_1_gru_cell_26_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ў
AssignVariableOp_4AssignVariableOp)assignvariableop_4_gru_1_gru_cell_26_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5А
AssignVariableOp_5AssignVariableOp+assignvariableop_5_gru_2_gru_cell_27_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6К
AssignVariableOp_6AssignVariableOp5assignvariableop_6_gru_2_gru_cell_27_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ў
AssignVariableOp_7AssignVariableOp)assignvariableop_7_gru_2_gru_cell_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ЇX
ѕ
A__inference_GRU_2_layer_call_and_return_conditional_losses_696651
inputs_0'
#gru_cell_27_readvariableop_resource.
*gru_cell_27_matmul_readvariableop_resource0
,gru_cell_27_matmul_1_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_27/ReadVariableOp
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_27/unstackБ
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_27/MatMul/ReadVariableOpЉ
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMulЃ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAddh
gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_27/Const
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split/split_dimм
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/splitЗ
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_27/MatMul_1/ReadVariableOpЅ
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMul_1Љ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAdd_1
gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_27/Const_1
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split_1/split_dim
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const_1:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/split_1
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add|
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_1
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid_1
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_2u
gru_cell_27/TanhTanhgru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Tanh
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_1k
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_27/sub/x
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/sub
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_2
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_696561*
condR
while_cond_696560*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
@
Е
while_body_696040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_26_readvariableop_resource_06
2while_gru_cell_26_matmul_readvariableop_resource_08
4while_gru_cell_26_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_26_readvariableop_resource4
0while_gru_cell_26_matmul_readvariableop_resource6
2while_gru_cell_26_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_26/ReadVariableOp 
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_26/unstackХ
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_26/MatMul/ReadVariableOpг
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMulЛ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAddt
while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_26/Const
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_26/split/split_dimє
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/splitЫ
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_26/MatMul_1/ReadVariableOpМ
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMul_1С
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAdd_1
while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_26/Const_1
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_26/split_1/split_dim­
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0"while/gru_cell_26/Const_1:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/split_1Џ
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/SigmoidГ
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_1
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Sigmoid_1Ќ
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mulЊ
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_2
while/gru_cell_26/TanhTanhwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Tanh
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_1w
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_26/sub/xЈ
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/subЂ
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_2Ї
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
@
Е
while_body_693902
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_27_readvariableop_resource_06
2while_gru_cell_27_matmul_readvariableop_resource_08
4while_gru_cell_27_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_27_readvariableop_resource4
0while_gru_cell_27_matmul_readvariableop_resource6
2while_gru_cell_27_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_27/ReadVariableOp 
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_27/unstackХ
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_27/MatMul/ReadVariableOpг
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMulЛ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAddt
while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_27/Const
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_27/split/split_dimє
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/splitЫ
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_27/MatMul_1/ReadVariableOpМ
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMul_1С
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAdd_1
while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_27/Const_1
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_27/split_1/split_dim­
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0"while/gru_cell_27/Const_1:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/split_1Џ
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/SigmoidГ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_1
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Sigmoid_1Ќ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mulЊ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_2
while/gru_cell_27/TanhTanhwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Tanh
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_1w
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_27/sub/xЈ
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/subЂ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_2Ї
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ЇX
ѕ
A__inference_GRU_1_layer_call_and_return_conditional_losses_695971
inputs_0'
#gru_cell_26_readvariableop_resource.
*gru_cell_26_matmul_readvariableop_resource0
,gru_cell_26_matmul_1_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_26/ReadVariableOp
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_26/unstackБ
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_26/MatMul/ReadVariableOpЉ
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMulЃ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAddh
gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_26/Const
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split/split_dimм
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/splitЗ
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_26/MatMul_1/ReadVariableOpЅ
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMul_1Љ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAdd_1
gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_26/Const_1
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split_1/split_dim
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const_1:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/split_1
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add|
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_1
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid_1
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_2u
gru_cell_26/TanhTanhgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Tanh
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_1k
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_26/sub/x
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/sub
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_2
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_695881*
condR
while_cond_695880*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ю
ы
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_697292

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ц
щ
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_692471

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Р
п
+__inference_Supervisor_layer_call_fn_695067
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Supervisor_layer_call_and_return_conditional_losses_6942782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameGRU_1_input
н
Њ
?__inference_OUT_layer_call_and_return_conditional_losses_697203

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
BiasAdde
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2	
Sigmoidc
IdentityIdentitySigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ:::S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
э
"Supervisor_GRU_2_while_cond_692241>
:supervisor_gru_2_while_supervisor_gru_2_while_loop_counterD
@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations&
"supervisor_gru_2_while_placeholder(
$supervisor_gru_2_while_placeholder_1(
$supervisor_gru_2_while_placeholder_2@
<supervisor_gru_2_while_less_supervisor_gru_2_strided_slice_1V
Rsupervisor_gru_2_while_supervisor_gru_2_while_cond_692241___redundant_placeholder0V
Rsupervisor_gru_2_while_supervisor_gru_2_while_cond_692241___redundant_placeholder1V
Rsupervisor_gru_2_while_supervisor_gru_2_while_cond_692241___redundant_placeholder2V
Rsupervisor_gru_2_while_supervisor_gru_2_while_cond_692241___redundant_placeholder3#
supervisor_gru_2_while_identity
Х
Supervisor/GRU_2/while/LessLess"supervisor_gru_2_while_placeholder<supervisor_gru_2_while_less_supervisor_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
Supervisor/GRU_2/while/Less
Supervisor/GRU_2/while/IdentityIdentitySupervisor/GRU_2/while/Less:z:0*
T0
*
_output_shapes
: 2!
Supervisor/GRU_2/while/Identity"K
supervisor_gru_2_while_identity(Supervisor/GRU_2/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ю!
л
while_body_693292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_27_693314_0
while_gru_cell_27_693316_0
while_gru_cell_27_693318_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_27_693314
while_gru_cell_27_693316
while_gru_cell_27_693318Ђ)while/gru_cell_27/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
)while/gru_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_27_693314_0while_gru_cell_27_693316_0while_gru_cell_27_693318_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_6929932+
)while/gru_cell_27/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_27/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_27/StatefulPartitionedCall:output:1*^while/gru_cell_27/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"6
while_gru_cell_27_693314while_gru_cell_27_693314_0"6
while_gru_cell_27_693316while_gru_cell_27_693316_0"6
while_gru_cell_27_693318while_gru_cell_27_693318_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_27/StatefulPartitionedCall)while/gru_cell_27/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
о	
Ў
,__inference_gru_cell_27_layer_call_fn_697428

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_6930332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
шW
ѓ
A__inference_GRU_2_layer_call_and_return_conditional_losses_694151

inputs'
#gru_cell_27_readvariableop_resource.
*gru_cell_27_matmul_readvariableop_resource0
,gru_cell_27_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_27/ReadVariableOp
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_27/unstackБ
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_27/MatMul/ReadVariableOpЉ
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMulЃ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAddh
gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_27/Const
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split/split_dimм
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/splitЗ
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_27/MatMul_1/ReadVariableOpЅ
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMul_1Љ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAdd_1
gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_27/Const_1
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split_1/split_dim
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const_1:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/split_1
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add|
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_1
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid_1
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_2u
gru_cell_27/TanhTanhgru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Tanh
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_1k
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_27/sub/x
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/sub
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_2
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_694061*
condR
while_cond_694060*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
н
!__inference__wrapped_model_692359
gru_1_input8
4supervisor_gru_1_gru_cell_26_readvariableop_resource?
;supervisor_gru_1_gru_cell_26_matmul_readvariableop_resourceA
=supervisor_gru_1_gru_cell_26_matmul_1_readvariableop_resource8
4supervisor_gru_2_gru_cell_27_readvariableop_resource?
;supervisor_gru_2_gru_cell_27_matmul_readvariableop_resourceA
=supervisor_gru_2_gru_cell_27_matmul_1_readvariableop_resource4
0supervisor_out_tensordot_readvariableop_resource2
.supervisor_out_biasadd_readvariableop_resource
identityЂSupervisor/GRU_1/whileЂSupervisor/GRU_2/whilek
Supervisor/GRU_1/ShapeShapegru_1_input*
T0*
_output_shapes
:2
Supervisor/GRU_1/Shape
$Supervisor/GRU_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Supervisor/GRU_1/strided_slice/stack
&Supervisor/GRU_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Supervisor/GRU_1/strided_slice/stack_1
&Supervisor/GRU_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Supervisor/GRU_1/strided_slice/stack_2Ш
Supervisor/GRU_1/strided_sliceStridedSliceSupervisor/GRU_1/Shape:output:0-Supervisor/GRU_1/strided_slice/stack:output:0/Supervisor/GRU_1/strided_slice/stack_1:output:0/Supervisor/GRU_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Supervisor/GRU_1/strided_slice~
Supervisor/GRU_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
Supervisor/GRU_1/zeros/mul/yА
Supervisor/GRU_1/zeros/mulMul'Supervisor/GRU_1/strided_slice:output:0%Supervisor/GRU_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_1/zeros/mul
Supervisor/GRU_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
Supervisor/GRU_1/zeros/Less/yЋ
Supervisor/GRU_1/zeros/LessLessSupervisor/GRU_1/zeros/mul:z:0&Supervisor/GRU_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_1/zeros/Less
Supervisor/GRU_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2!
Supervisor/GRU_1/zeros/packed/1Ч
Supervisor/GRU_1/zeros/packedPack'Supervisor/GRU_1/strided_slice:output:0(Supervisor/GRU_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
Supervisor/GRU_1/zeros/packed
Supervisor/GRU_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Supervisor/GRU_1/zeros/ConstЙ
Supervisor/GRU_1/zerosFill&Supervisor/GRU_1/zeros/packed:output:0%Supervisor/GRU_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Supervisor/GRU_1/zeros
Supervisor/GRU_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
Supervisor/GRU_1/transpose/permВ
Supervisor/GRU_1/transpose	Transposegru_1_input(Supervisor/GRU_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/GRU_1/transpose
Supervisor/GRU_1/Shape_1ShapeSupervisor/GRU_1/transpose:y:0*
T0*
_output_shapes
:2
Supervisor/GRU_1/Shape_1
&Supervisor/GRU_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Supervisor/GRU_1/strided_slice_1/stack
(Supervisor/GRU_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_1/strided_slice_1/stack_1
(Supervisor/GRU_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_1/strided_slice_1/stack_2д
 Supervisor/GRU_1/strided_slice_1StridedSlice!Supervisor/GRU_1/Shape_1:output:0/Supervisor/GRU_1/strided_slice_1/stack:output:01Supervisor/GRU_1/strided_slice_1/stack_1:output:01Supervisor/GRU_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Supervisor/GRU_1/strided_slice_1Ї
,Supervisor/GRU_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_1/TensorArrayV2/element_shapeі
Supervisor/GRU_1/TensorArrayV2TensorListReserve5Supervisor/GRU_1/TensorArrayV2/element_shape:output:0)Supervisor/GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
Supervisor/GRU_1/TensorArrayV2с
FSupervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
FSupervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8Supervisor/GRU_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorSupervisor/GRU_1/transpose:y:0OSupervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8Supervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor
&Supervisor/GRU_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Supervisor/GRU_1/strided_slice_2/stack
(Supervisor/GRU_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_1/strided_slice_2/stack_1
(Supervisor/GRU_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_1/strided_slice_2/stack_2т
 Supervisor/GRU_1/strided_slice_2StridedSliceSupervisor/GRU_1/transpose:y:0/Supervisor/GRU_1/strided_slice_2/stack:output:01Supervisor/GRU_1/strided_slice_2/stack_1:output:01Supervisor/GRU_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 Supervisor/GRU_1/strided_slice_2Я
+Supervisor/GRU_1/gru_cell_26/ReadVariableOpReadVariableOp4supervisor_gru_1_gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02-
+Supervisor/GRU_1/gru_cell_26/ReadVariableOpС
$Supervisor/GRU_1/gru_cell_26/unstackUnpack3Supervisor/GRU_1/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2&
$Supervisor/GRU_1/gru_cell_26/unstackф
2Supervisor/GRU_1/gru_cell_26/MatMul/ReadVariableOpReadVariableOp;supervisor_gru_1_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype024
2Supervisor/GRU_1/gru_cell_26/MatMul/ReadVariableOpэ
#Supervisor/GRU_1/gru_cell_26/MatMulMatMul)Supervisor/GRU_1/strided_slice_2:output:0:Supervisor/GRU_1/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2%
#Supervisor/GRU_1/gru_cell_26/MatMulч
$Supervisor/GRU_1/gru_cell_26/BiasAddBiasAdd-Supervisor/GRU_1/gru_cell_26/MatMul:product:0-Supervisor/GRU_1/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2&
$Supervisor/GRU_1/gru_cell_26/BiasAdd
"Supervisor/GRU_1/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"Supervisor/GRU_1/gru_cell_26/ConstЇ
,Supervisor/GRU_1/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_1/gru_cell_26/split/split_dim 
"Supervisor/GRU_1/gru_cell_26/splitSplit5Supervisor/GRU_1/gru_cell_26/split/split_dim:output:0-Supervisor/GRU_1/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2$
"Supervisor/GRU_1/gru_cell_26/splitъ
4Supervisor/GRU_1/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp=supervisor_gru_1_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype026
4Supervisor/GRU_1/gru_cell_26/MatMul_1/ReadVariableOpщ
%Supervisor/GRU_1/gru_cell_26/MatMul_1MatMulSupervisor/GRU_1/zeros:output:0<Supervisor/GRU_1/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2'
%Supervisor/GRU_1/gru_cell_26/MatMul_1э
&Supervisor/GRU_1/gru_cell_26/BiasAdd_1BiasAdd/Supervisor/GRU_1/gru_cell_26/MatMul_1:product:0-Supervisor/GRU_1/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2(
&Supervisor/GRU_1/gru_cell_26/BiasAdd_1Ё
$Supervisor/GRU_1/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2&
$Supervisor/GRU_1/gru_cell_26/Const_1Ћ
.Supervisor/GRU_1/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.Supervisor/GRU_1/gru_cell_26/split_1/split_dimф
$Supervisor/GRU_1/gru_cell_26/split_1SplitV/Supervisor/GRU_1/gru_cell_26/BiasAdd_1:output:0-Supervisor/GRU_1/gru_cell_26/Const_1:output:07Supervisor/GRU_1/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2&
$Supervisor/GRU_1/gru_cell_26/split_1л
 Supervisor/GRU_1/gru_cell_26/addAddV2+Supervisor/GRU_1/gru_cell_26/split:output:0-Supervisor/GRU_1/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_26/addЏ
$Supervisor/GRU_1/gru_cell_26/SigmoidSigmoid$Supervisor/GRU_1/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$Supervisor/GRU_1/gru_cell_26/Sigmoidп
"Supervisor/GRU_1/gru_cell_26/add_1AddV2+Supervisor/GRU_1/gru_cell_26/split:output:1-Supervisor/GRU_1/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_26/add_1Е
&Supervisor/GRU_1/gru_cell_26/Sigmoid_1Sigmoid&Supervisor/GRU_1/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/gru_cell_26/Sigmoid_1и
 Supervisor/GRU_1/gru_cell_26/mulMul*Supervisor/GRU_1/gru_cell_26/Sigmoid_1:y:0-Supervisor/GRU_1/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_26/mulж
"Supervisor/GRU_1/gru_cell_26/add_2AddV2+Supervisor/GRU_1/gru_cell_26/split:output:2$Supervisor/GRU_1/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_26/add_2Ј
!Supervisor/GRU_1/gru_cell_26/TanhTanh&Supervisor/GRU_1/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_1/gru_cell_26/TanhЬ
"Supervisor/GRU_1/gru_cell_26/mul_1Mul(Supervisor/GRU_1/gru_cell_26/Sigmoid:y:0Supervisor/GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_26/mul_1
"Supervisor/GRU_1/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"Supervisor/GRU_1/gru_cell_26/sub/xд
 Supervisor/GRU_1/gru_cell_26/subSub+Supervisor/GRU_1/gru_cell_26/sub/x:output:0(Supervisor/GRU_1/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_26/subЮ
"Supervisor/GRU_1/gru_cell_26/mul_2Mul$Supervisor/GRU_1/gru_cell_26/sub:z:0%Supervisor/GRU_1/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_26/mul_2г
"Supervisor/GRU_1/gru_cell_26/add_3AddV2&Supervisor/GRU_1/gru_cell_26/mul_1:z:0&Supervisor/GRU_1/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_26/add_3Б
.Supervisor/GRU_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   20
.Supervisor/GRU_1/TensorArrayV2_1/element_shapeќ
 Supervisor/GRU_1/TensorArrayV2_1TensorListReserve7Supervisor/GRU_1/TensorArrayV2_1/element_shape:output:0)Supervisor/GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 Supervisor/GRU_1/TensorArrayV2_1p
Supervisor/GRU_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
Supervisor/GRU_1/timeЁ
)Supervisor/GRU_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)Supervisor/GRU_1/while/maximum_iterations
#Supervisor/GRU_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#Supervisor/GRU_1/while/loop_counter
Supervisor/GRU_1/whileWhile,Supervisor/GRU_1/while/loop_counter:output:02Supervisor/GRU_1/while/maximum_iterations:output:0Supervisor/GRU_1/time:output:0)Supervisor/GRU_1/TensorArrayV2_1:handle:0Supervisor/GRU_1/zeros:output:0)Supervisor/GRU_1/strided_slice_1:output:0HSupervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:04supervisor_gru_1_gru_cell_26_readvariableop_resource;supervisor_gru_1_gru_cell_26_matmul_readvariableop_resource=supervisor_gru_1_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*.
body&R$
"Supervisor_GRU_1_while_body_692087*.
cond&R$
"Supervisor_GRU_1_while_cond_692086*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
Supervisor/GRU_1/whileз
ASupervisor/GRU_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
ASupervisor/GRU_1/TensorArrayV2Stack/TensorListStack/element_shapeЌ
3Supervisor/GRU_1/TensorArrayV2Stack/TensorListStackTensorListStackSupervisor/GRU_1/while:output:3JSupervisor/GRU_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype025
3Supervisor/GRU_1/TensorArrayV2Stack/TensorListStackЃ
&Supervisor/GRU_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&Supervisor/GRU_1/strided_slice_3/stack
(Supervisor/GRU_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(Supervisor/GRU_1/strided_slice_3/stack_1
(Supervisor/GRU_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_1/strided_slice_3/stack_2
 Supervisor/GRU_1/strided_slice_3StridedSlice<Supervisor/GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0/Supervisor/GRU_1/strided_slice_3/stack:output:01Supervisor/GRU_1/strided_slice_3/stack_1:output:01Supervisor/GRU_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 Supervisor/GRU_1/strided_slice_3
!Supervisor/GRU_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!Supervisor/GRU_1/transpose_1/permщ
Supervisor/GRU_1/transpose_1	Transpose<Supervisor/GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0*Supervisor/GRU_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/GRU_1/transpose_1
Supervisor/GRU_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
Supervisor/GRU_1/runtime
Supervisor/GRU_2/ShapeShape Supervisor/GRU_1/transpose_1:y:0*
T0*
_output_shapes
:2
Supervisor/GRU_2/Shape
$Supervisor/GRU_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Supervisor/GRU_2/strided_slice/stack
&Supervisor/GRU_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Supervisor/GRU_2/strided_slice/stack_1
&Supervisor/GRU_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Supervisor/GRU_2/strided_slice/stack_2Ш
Supervisor/GRU_2/strided_sliceStridedSliceSupervisor/GRU_2/Shape:output:0-Supervisor/GRU_2/strided_slice/stack:output:0/Supervisor/GRU_2/strided_slice/stack_1:output:0/Supervisor/GRU_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Supervisor/GRU_2/strided_slice~
Supervisor/GRU_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
Supervisor/GRU_2/zeros/mul/yА
Supervisor/GRU_2/zeros/mulMul'Supervisor/GRU_2/strided_slice:output:0%Supervisor/GRU_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_2/zeros/mul
Supervisor/GRU_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
Supervisor/GRU_2/zeros/Less/yЋ
Supervisor/GRU_2/zeros/LessLessSupervisor/GRU_2/zeros/mul:z:0&Supervisor/GRU_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_2/zeros/Less
Supervisor/GRU_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2!
Supervisor/GRU_2/zeros/packed/1Ч
Supervisor/GRU_2/zeros/packedPack'Supervisor/GRU_2/strided_slice:output:0(Supervisor/GRU_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
Supervisor/GRU_2/zeros/packed
Supervisor/GRU_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Supervisor/GRU_2/zeros/ConstЙ
Supervisor/GRU_2/zerosFill&Supervisor/GRU_2/zeros/packed:output:0%Supervisor/GRU_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Supervisor/GRU_2/zeros
Supervisor/GRU_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
Supervisor/GRU_2/transpose/permЧ
Supervisor/GRU_2/transpose	Transpose Supervisor/GRU_1/transpose_1:y:0(Supervisor/GRU_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/GRU_2/transpose
Supervisor/GRU_2/Shape_1ShapeSupervisor/GRU_2/transpose:y:0*
T0*
_output_shapes
:2
Supervisor/GRU_2/Shape_1
&Supervisor/GRU_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Supervisor/GRU_2/strided_slice_1/stack
(Supervisor/GRU_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_2/strided_slice_1/stack_1
(Supervisor/GRU_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_2/strided_slice_1/stack_2д
 Supervisor/GRU_2/strided_slice_1StridedSlice!Supervisor/GRU_2/Shape_1:output:0/Supervisor/GRU_2/strided_slice_1/stack:output:01Supervisor/GRU_2/strided_slice_1/stack_1:output:01Supervisor/GRU_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Supervisor/GRU_2/strided_slice_1Ї
,Supervisor/GRU_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_2/TensorArrayV2/element_shapeі
Supervisor/GRU_2/TensorArrayV2TensorListReserve5Supervisor/GRU_2/TensorArrayV2/element_shape:output:0)Supervisor/GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
Supervisor/GRU_2/TensorArrayV2с
FSupervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2H
FSupervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shapeМ
8Supervisor/GRU_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorSupervisor/GRU_2/transpose:y:0OSupervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8Supervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor
&Supervisor/GRU_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Supervisor/GRU_2/strided_slice_2/stack
(Supervisor/GRU_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_2/strided_slice_2/stack_1
(Supervisor/GRU_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_2/strided_slice_2/stack_2т
 Supervisor/GRU_2/strided_slice_2StridedSliceSupervisor/GRU_2/transpose:y:0/Supervisor/GRU_2/strided_slice_2/stack:output:01Supervisor/GRU_2/strided_slice_2/stack_1:output:01Supervisor/GRU_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 Supervisor/GRU_2/strided_slice_2Я
+Supervisor/GRU_2/gru_cell_27/ReadVariableOpReadVariableOp4supervisor_gru_2_gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02-
+Supervisor/GRU_2/gru_cell_27/ReadVariableOpС
$Supervisor/GRU_2/gru_cell_27/unstackUnpack3Supervisor/GRU_2/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2&
$Supervisor/GRU_2/gru_cell_27/unstackф
2Supervisor/GRU_2/gru_cell_27/MatMul/ReadVariableOpReadVariableOp;supervisor_gru_2_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype024
2Supervisor/GRU_2/gru_cell_27/MatMul/ReadVariableOpэ
#Supervisor/GRU_2/gru_cell_27/MatMulMatMul)Supervisor/GRU_2/strided_slice_2:output:0:Supervisor/GRU_2/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2%
#Supervisor/GRU_2/gru_cell_27/MatMulч
$Supervisor/GRU_2/gru_cell_27/BiasAddBiasAdd-Supervisor/GRU_2/gru_cell_27/MatMul:product:0-Supervisor/GRU_2/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2&
$Supervisor/GRU_2/gru_cell_27/BiasAdd
"Supervisor/GRU_2/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"Supervisor/GRU_2/gru_cell_27/ConstЇ
,Supervisor/GRU_2/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_2/gru_cell_27/split/split_dim 
"Supervisor/GRU_2/gru_cell_27/splitSplit5Supervisor/GRU_2/gru_cell_27/split/split_dim:output:0-Supervisor/GRU_2/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2$
"Supervisor/GRU_2/gru_cell_27/splitъ
4Supervisor/GRU_2/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp=supervisor_gru_2_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype026
4Supervisor/GRU_2/gru_cell_27/MatMul_1/ReadVariableOpщ
%Supervisor/GRU_2/gru_cell_27/MatMul_1MatMulSupervisor/GRU_2/zeros:output:0<Supervisor/GRU_2/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2'
%Supervisor/GRU_2/gru_cell_27/MatMul_1э
&Supervisor/GRU_2/gru_cell_27/BiasAdd_1BiasAdd/Supervisor/GRU_2/gru_cell_27/MatMul_1:product:0-Supervisor/GRU_2/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2(
&Supervisor/GRU_2/gru_cell_27/BiasAdd_1Ё
$Supervisor/GRU_2/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2&
$Supervisor/GRU_2/gru_cell_27/Const_1Ћ
.Supervisor/GRU_2/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.Supervisor/GRU_2/gru_cell_27/split_1/split_dimф
$Supervisor/GRU_2/gru_cell_27/split_1SplitV/Supervisor/GRU_2/gru_cell_27/BiasAdd_1:output:0-Supervisor/GRU_2/gru_cell_27/Const_1:output:07Supervisor/GRU_2/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2&
$Supervisor/GRU_2/gru_cell_27/split_1л
 Supervisor/GRU_2/gru_cell_27/addAddV2+Supervisor/GRU_2/gru_cell_27/split:output:0-Supervisor/GRU_2/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_27/addЏ
$Supervisor/GRU_2/gru_cell_27/SigmoidSigmoid$Supervisor/GRU_2/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$Supervisor/GRU_2/gru_cell_27/Sigmoidп
"Supervisor/GRU_2/gru_cell_27/add_1AddV2+Supervisor/GRU_2/gru_cell_27/split:output:1-Supervisor/GRU_2/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_27/add_1Е
&Supervisor/GRU_2/gru_cell_27/Sigmoid_1Sigmoid&Supervisor/GRU_2/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/gru_cell_27/Sigmoid_1и
 Supervisor/GRU_2/gru_cell_27/mulMul*Supervisor/GRU_2/gru_cell_27/Sigmoid_1:y:0-Supervisor/GRU_2/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_27/mulж
"Supervisor/GRU_2/gru_cell_27/add_2AddV2+Supervisor/GRU_2/gru_cell_27/split:output:2$Supervisor/GRU_2/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_27/add_2Ј
!Supervisor/GRU_2/gru_cell_27/TanhTanh&Supervisor/GRU_2/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_2/gru_cell_27/TanhЬ
"Supervisor/GRU_2/gru_cell_27/mul_1Mul(Supervisor/GRU_2/gru_cell_27/Sigmoid:y:0Supervisor/GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_27/mul_1
"Supervisor/GRU_2/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"Supervisor/GRU_2/gru_cell_27/sub/xд
 Supervisor/GRU_2/gru_cell_27/subSub+Supervisor/GRU_2/gru_cell_27/sub/x:output:0(Supervisor/GRU_2/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_27/subЮ
"Supervisor/GRU_2/gru_cell_27/mul_2Mul$Supervisor/GRU_2/gru_cell_27/sub:z:0%Supervisor/GRU_2/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_27/mul_2г
"Supervisor/GRU_2/gru_cell_27/add_3AddV2&Supervisor/GRU_2/gru_cell_27/mul_1:z:0&Supervisor/GRU_2/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_27/add_3Б
.Supervisor/GRU_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   20
.Supervisor/GRU_2/TensorArrayV2_1/element_shapeќ
 Supervisor/GRU_2/TensorArrayV2_1TensorListReserve7Supervisor/GRU_2/TensorArrayV2_1/element_shape:output:0)Supervisor/GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 Supervisor/GRU_2/TensorArrayV2_1p
Supervisor/GRU_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
Supervisor/GRU_2/timeЁ
)Supervisor/GRU_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)Supervisor/GRU_2/while/maximum_iterations
#Supervisor/GRU_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#Supervisor/GRU_2/while/loop_counter
Supervisor/GRU_2/whileWhile,Supervisor/GRU_2/while/loop_counter:output:02Supervisor/GRU_2/while/maximum_iterations:output:0Supervisor/GRU_2/time:output:0)Supervisor/GRU_2/TensorArrayV2_1:handle:0Supervisor/GRU_2/zeros:output:0)Supervisor/GRU_2/strided_slice_1:output:0HSupervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:04supervisor_gru_2_gru_cell_27_readvariableop_resource;supervisor_gru_2_gru_cell_27_matmul_readvariableop_resource=supervisor_gru_2_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*.
body&R$
"Supervisor_GRU_2_while_body_692242*.
cond&R$
"Supervisor_GRU_2_while_cond_692241*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
Supervisor/GRU_2/whileз
ASupervisor/GRU_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
ASupervisor/GRU_2/TensorArrayV2Stack/TensorListStack/element_shapeЌ
3Supervisor/GRU_2/TensorArrayV2Stack/TensorListStackTensorListStackSupervisor/GRU_2/while:output:3JSupervisor/GRU_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype025
3Supervisor/GRU_2/TensorArrayV2Stack/TensorListStackЃ
&Supervisor/GRU_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2(
&Supervisor/GRU_2/strided_slice_3/stack
(Supervisor/GRU_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(Supervisor/GRU_2/strided_slice_3/stack_1
(Supervisor/GRU_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Supervisor/GRU_2/strided_slice_3/stack_2
 Supervisor/GRU_2/strided_slice_3StridedSlice<Supervisor/GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0/Supervisor/GRU_2/strided_slice_3/stack:output:01Supervisor/GRU_2/strided_slice_3/stack_1:output:01Supervisor/GRU_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2"
 Supervisor/GRU_2/strided_slice_3
!Supervisor/GRU_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!Supervisor/GRU_2/transpose_1/permщ
Supervisor/GRU_2/transpose_1	Transpose<Supervisor/GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0*Supervisor/GRU_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/GRU_2/transpose_1
Supervisor/GRU_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
Supervisor/GRU_2/runtimeУ
'Supervisor/OUT/Tensordot/ReadVariableOpReadVariableOp0supervisor_out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02)
'Supervisor/OUT/Tensordot/ReadVariableOp
Supervisor/OUT/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Supervisor/OUT/Tensordot/axes
Supervisor/OUT/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Supervisor/OUT/Tensordot/free
Supervisor/OUT/Tensordot/ShapeShape Supervisor/GRU_2/transpose_1:y:0*
T0*
_output_shapes
:2 
Supervisor/OUT/Tensordot/Shape
&Supervisor/OUT/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&Supervisor/OUT/Tensordot/GatherV2/axis
!Supervisor/OUT/Tensordot/GatherV2GatherV2'Supervisor/OUT/Tensordot/Shape:output:0&Supervisor/OUT/Tensordot/free:output:0/Supervisor/OUT/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!Supervisor/OUT/Tensordot/GatherV2
(Supervisor/OUT/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(Supervisor/OUT/Tensordot/GatherV2_1/axisЂ
#Supervisor/OUT/Tensordot/GatherV2_1GatherV2'Supervisor/OUT/Tensordot/Shape:output:0&Supervisor/OUT/Tensordot/axes:output:01Supervisor/OUT/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#Supervisor/OUT/Tensordot/GatherV2_1
Supervisor/OUT/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
Supervisor/OUT/Tensordot/ConstМ
Supervisor/OUT/Tensordot/ProdProd*Supervisor/OUT/Tensordot/GatherV2:output:0'Supervisor/OUT/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Supervisor/OUT/Tensordot/Prod
 Supervisor/OUT/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 Supervisor/OUT/Tensordot/Const_1Ф
Supervisor/OUT/Tensordot/Prod_1Prod,Supervisor/OUT/Tensordot/GatherV2_1:output:0)Supervisor/OUT/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2!
Supervisor/OUT/Tensordot/Prod_1
$Supervisor/OUT/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Supervisor/OUT/Tensordot/concat/axisћ
Supervisor/OUT/Tensordot/concatConcatV2&Supervisor/OUT/Tensordot/free:output:0&Supervisor/OUT/Tensordot/axes:output:0-Supervisor/OUT/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2!
Supervisor/OUT/Tensordot/concatШ
Supervisor/OUT/Tensordot/stackPack&Supervisor/OUT/Tensordot/Prod:output:0(Supervisor/OUT/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2 
Supervisor/OUT/Tensordot/stackз
"Supervisor/OUT/Tensordot/transpose	Transpose Supervisor/GRU_2/transpose_1:y:0(Supervisor/OUT/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2$
"Supervisor/OUT/Tensordot/transposeл
 Supervisor/OUT/Tensordot/ReshapeReshape&Supervisor/OUT/Tensordot/transpose:y:0'Supervisor/OUT/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2"
 Supervisor/OUT/Tensordot/Reshapeк
Supervisor/OUT/Tensordot/MatMulMatMul)Supervisor/OUT/Tensordot/Reshape:output:0/Supervisor/OUT/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
Supervisor/OUT/Tensordot/MatMul
 Supervisor/OUT/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Supervisor/OUT/Tensordot/Const_2
&Supervisor/OUT/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&Supervisor/OUT/Tensordot/concat_1/axis
!Supervisor/OUT/Tensordot/concat_1ConcatV2*Supervisor/OUT/Tensordot/GatherV2:output:0)Supervisor/OUT/Tensordot/Const_2:output:0/Supervisor/OUT/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2#
!Supervisor/OUT/Tensordot/concat_1Ь
Supervisor/OUT/TensordotReshape)Supervisor/OUT/Tensordot/MatMul:product:0*Supervisor/OUT/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/OUT/TensordotЙ
%Supervisor/OUT/BiasAdd/ReadVariableOpReadVariableOp.supervisor_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Supervisor/OUT/BiasAdd/ReadVariableOpУ
Supervisor/OUT/BiasAddBiasAdd!Supervisor/OUT/Tensordot:output:0-Supervisor/OUT/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/OUT/BiasAdd
Supervisor/OUT/SigmoidSigmoidSupervisor/OUT/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Supervisor/OUT/SigmoidЄ
IdentityIdentitySupervisor/OUT/Sigmoid:y:0^Supervisor/GRU_1/while^Supervisor/GRU_2/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::20
Supervisor/GRU_1/whileSupervisor/GRU_1/while20
Supervisor/GRU_2/whileSupervisor/GRU_2/while:X T
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameGRU_1_input
а
Њ
while_cond_693901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_693901___redundant_placeholder04
0while_while_cond_693901___redundant_placeholder14
0while_while_cond_693901___redundant_placeholder24
0while_while_cond_693901___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
@
Е
while_body_696720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_27_readvariableop_resource_06
2while_gru_cell_27_matmul_readvariableop_resource_08
4while_gru_cell_27_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_27_readvariableop_resource4
0while_gru_cell_27_matmul_readvariableop_resource6
2while_gru_cell_27_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_27/ReadVariableOp 
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_27/unstackХ
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_27/MatMul/ReadVariableOpг
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMulЛ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAddt
while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_27/Const
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_27/split/split_dimє
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/splitЫ
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_27/MatMul_1/ReadVariableOpМ
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMul_1С
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAdd_1
while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_27/Const_1
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_27/split_1/split_dim­
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0"while/gru_cell_27/Const_1:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/split_1Џ
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/SigmoidГ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_1
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Sigmoid_1Ќ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mulЊ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_2
while/gru_cell_27/TanhTanhwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Tanh
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_1w
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_27/sub/xЈ
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/subЂ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_2Ї
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ю!
л
while_body_692848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_26_692870_0
while_gru_cell_26_692872_0
while_gru_cell_26_692874_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_26_692870
while_gru_cell_26_692872
while_gru_cell_26_692874Ђ)while/gru_cell_26/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemБ
)while/gru_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_26_692870_0while_gru_cell_26_692872_0while_gru_cell_26_692874_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_6924712+
)while/gru_cell_26/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_26/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_26/StatefulPartitionedCall:output:1*^while/gru_cell_26/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"6
while_gru_cell_26_692870while_gru_cell_26_692870_0"6
while_gru_cell_26_692872while_gru_cell_26_692872_0"6
while_gru_cell_26_692874while_gru_cell_26_692874_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_26/StatefulPartitionedCall)while/gru_cell_26/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
а
Њ
while_cond_696560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_696560___redundant_placeholder04
0while_while_cond_696560___redundant_placeholder14
0while_while_cond_696560___redundant_placeholder24
0while_while_cond_696560___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ѕ
Я
F__inference_Supervisor_layer_call_and_return_conditional_losses_694278

inputs
gru_1_694258
gru_1_694260
gru_1_694262
gru_2_694265
gru_2_694267
gru_2_694269

out_694272

out_694274
identityЂGRU_1/StatefulPartitionedCallЂGRU_2/StatefulPartitionedCallЂOUT/StatefulPartitionedCall
GRU_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_694258gru_1_694260gru_1_694262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_1_layer_call_and_return_conditional_losses_6936452
GRU_1/StatefulPartitionedCallЙ
GRU_2/StatefulPartitionedCallStatefulPartitionedCall&GRU_1/StatefulPartitionedCall:output:0gru_2_694265gru_2_694267gru_2_694269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_2_layer_call_and_return_conditional_losses_6939922
GRU_2/StatefulPartitionedCall
OUT/StatefulPartitionedCallStatefulPartitionedCall&GRU_2/StatefulPartitionedCall:output:0
out_694272
out_694274*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_OUT_layer_call_and_return_conditional_losses_6942122
OUT/StatefulPartitionedCallк
IdentityIdentity$OUT/StatefulPartitionedCall:output:0^GRU_1/StatefulPartitionedCall^GRU_2/StatefulPartitionedCall^OUT/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::2>
GRU_1/StatefulPartitionedCallGRU_1/StatefulPartitionedCall2>
GRU_2/StatefulPartitionedCallGRU_2/StatefulPartitionedCall2:
OUT/StatefulPartitionedCallOUT/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
э
"Supervisor_GRU_1_while_cond_692086>
:supervisor_gru_1_while_supervisor_gru_1_while_loop_counterD
@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations&
"supervisor_gru_1_while_placeholder(
$supervisor_gru_1_while_placeholder_1(
$supervisor_gru_1_while_placeholder_2@
<supervisor_gru_1_while_less_supervisor_gru_1_strided_slice_1V
Rsupervisor_gru_1_while_supervisor_gru_1_while_cond_692086___redundant_placeholder0V
Rsupervisor_gru_1_while_supervisor_gru_1_while_cond_692086___redundant_placeholder1V
Rsupervisor_gru_1_while_supervisor_gru_1_while_cond_692086___redundant_placeholder2V
Rsupervisor_gru_1_while_supervisor_gru_1_while_cond_692086___redundant_placeholder3#
supervisor_gru_1_while_identity
Х
Supervisor/GRU_1/while/LessLess"supervisor_gru_1_while_placeholder<supervisor_gru_1_while_less_supervisor_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
Supervisor/GRU_1/while/Less
Supervisor/GRU_1/while/IdentityIdentitySupervisor/GRU_1/while/Less:z:0*
T0
*
_output_shapes
: 2!
Supervisor/GRU_1/while/Identity"K
supervisor_gru_1_while_identity(Supervisor/GRU_1/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:


&__inference_GRU_2_layer_call_fn_697172

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_2_layer_call_and_return_conditional_losses_6941512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
@
Е
while_body_696561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_27_readvariableop_resource_06
2while_gru_cell_27_matmul_readvariableop_resource_08
4while_gru_cell_27_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_27_readvariableop_resource4
0while_gru_cell_27_matmul_readvariableop_resource6
2while_gru_cell_27_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_27/ReadVariableOp 
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_27/unstackХ
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_27/MatMul/ReadVariableOpг
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMulЛ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAddt
while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_27/Const
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_27/split/split_dimє
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/splitЫ
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_27/MatMul_1/ReadVariableOpМ
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMul_1С
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAdd_1
while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_27/Const_1
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_27/split_1/split_dim­
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0"while/gru_cell_27/Const_1:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/split_1Џ
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/SigmoidГ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_1
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Sigmoid_1Ќ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mulЊ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_2
while/gru_cell_27/TanhTanhwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Tanh
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_1w
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_27/sub/xЈ
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/subЂ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_2Ї
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
ос

F__inference_Supervisor_layer_call_and_return_conditional_losses_695046
gru_1_input-
)gru_1_gru_cell_26_readvariableop_resource4
0gru_1_gru_cell_26_matmul_readvariableop_resource6
2gru_1_gru_cell_26_matmul_1_readvariableop_resource-
)gru_2_gru_cell_27_readvariableop_resource4
0gru_2_gru_cell_27_matmul_readvariableop_resource6
2gru_2_gru_cell_27_matmul_1_readvariableop_resource)
%out_tensordot_readvariableop_resource'
#out_biasadd_readvariableop_resource
identityЂGRU_1/whileЂGRU_2/whileU
GRU_1/ShapeShapegru_1_input*
T0*
_output_shapes
:2
GRU_1/Shape
GRU_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice/stack
GRU_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_1
GRU_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_2
GRU_1/strided_sliceStridedSliceGRU_1/Shape:output:0"GRU_1/strided_slice/stack:output:0$GRU_1/strided_slice/stack_1:output:0$GRU_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_sliceh
GRU_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/mul/y
GRU_1/zeros/mulMulGRU_1/strided_slice:output:0GRU_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/mulk
GRU_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_1/zeros/Less/y
GRU_1/zeros/LessLessGRU_1/zeros/mul:z:0GRU_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/Lessn
GRU_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/packed/1
GRU_1/zeros/packedPackGRU_1/strided_slice:output:0GRU_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_1/zeros/packedk
GRU_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/zeros/Const
GRU_1/zerosFillGRU_1/zeros/packed:output:0GRU_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/zeros
GRU_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose/perm
GRU_1/transpose	Transposegru_1_inputGRU_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transposea
GRU_1/Shape_1ShapeGRU_1/transpose:y:0*
T0*
_output_shapes
:2
GRU_1/Shape_1
GRU_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_1/stack
GRU_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_1
GRU_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_2
GRU_1/strided_slice_1StridedSliceGRU_1/Shape_1:output:0$GRU_1/strided_slice_1/stack:output:0&GRU_1/strided_slice_1/stack_1:output:0&GRU_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_slice_1
!GRU_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/TensorArrayV2/element_shapeЪ
GRU_1/TensorArrayV2TensorListReserve*GRU_1/TensorArrayV2/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2Ы
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_1/transpose:y:0DGRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_1/TensorArrayUnstack/TensorListFromTensor
GRU_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_2/stack
GRU_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_1
GRU_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_2 
GRU_1/strided_slice_2StridedSliceGRU_1/transpose:y:0$GRU_1/strided_slice_2/stack:output:0&GRU_1/strided_slice_2/stack_1:output:0&GRU_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_2Ў
 GRU_1/gru_cell_26/ReadVariableOpReadVariableOp)gru_1_gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_26/ReadVariableOp 
GRU_1/gru_cell_26/unstackUnpack(GRU_1/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_26/unstackУ
'GRU_1/gru_cell_26/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_26/MatMul/ReadVariableOpС
GRU_1/gru_cell_26/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMulЛ
GRU_1/gru_cell_26/BiasAddBiasAdd"GRU_1/gru_cell_26/MatMul:product:0"GRU_1/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAddt
GRU_1/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_26/Const
!GRU_1/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_26/split/split_dimє
GRU_1/gru_cell_26/splitSplit*GRU_1/gru_cell_26/split/split_dim:output:0"GRU_1/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/splitЩ
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_26/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMul_1С
GRU_1/gru_cell_26/BiasAdd_1BiasAdd$GRU_1/gru_cell_26/MatMul_1:product:0"GRU_1/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAdd_1
GRU_1/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_26/Const_1
#GRU_1/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_26/split_1/split_dim­
GRU_1/gru_cell_26/split_1SplitV$GRU_1/gru_cell_26/BiasAdd_1:output:0"GRU_1/gru_cell_26/Const_1:output:0,GRU_1/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/split_1Џ
GRU_1/gru_cell_26/addAddV2 GRU_1/gru_cell_26/split:output:0"GRU_1/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add
GRU_1/gru_cell_26/SigmoidSigmoidGRU_1/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/SigmoidГ
GRU_1/gru_cell_26/add_1AddV2 GRU_1/gru_cell_26/split:output:1"GRU_1/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_1
GRU_1/gru_cell_26/Sigmoid_1SigmoidGRU_1/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Sigmoid_1Ќ
GRU_1/gru_cell_26/mulMulGRU_1/gru_cell_26/Sigmoid_1:y:0"GRU_1/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mulЊ
GRU_1/gru_cell_26/add_2AddV2 GRU_1/gru_cell_26/split:output:2GRU_1/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_2
GRU_1/gru_cell_26/TanhTanhGRU_1/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Tanh 
GRU_1/gru_cell_26/mul_1MulGRU_1/gru_cell_26/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_1w
GRU_1/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_26/sub/xЈ
GRU_1/gru_cell_26/subSub GRU_1/gru_cell_26/sub/x:output:0GRU_1/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/subЂ
GRU_1/gru_cell_26/mul_2MulGRU_1/gru_cell_26/sub:z:0GRU_1/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_2Ї
GRU_1/gru_cell_26/add_3AddV2GRU_1/gru_cell_26/mul_1:z:0GRU_1/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_3
#GRU_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_1/TensorArrayV2_1/element_shapeа
GRU_1/TensorArrayV2_1TensorListReserve,GRU_1/TensorArrayV2_1/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2_1Z

GRU_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_1/time
GRU_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_1/while/maximum_iterationsv
GRU_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_1/while/loop_counterў
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_26_readvariableop_resource0gru_1_gru_cell_26_matmul_readvariableop_resource2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_1_while_body_694774*#
condR
GRU_1_while_cond_694773*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_1/whileС
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_1/TensorArrayV2Stack/TensorListStackTensorListStackGRU_1/while:output:3?GRU_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_1/TensorArrayV2Stack/TensorListStack
GRU_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_1/strided_slice_3/stack
GRU_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_3/stack_1
GRU_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_3/stack_2О
GRU_1/strided_slice_3StridedSlice1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_1/strided_slice_3/stack:output:0&GRU_1/strided_slice_3/stack_1:output:0&GRU_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_3
GRU_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose_1/permН
GRU_1/transpose_1	Transpose1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0GRU_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transpose_1r
GRU_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/runtime_
GRU_2/ShapeShapeGRU_1/transpose_1:y:0*
T0*
_output_shapes
:2
GRU_2/Shape
GRU_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice/stack
GRU_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_1
GRU_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_2
GRU_2/strided_sliceStridedSliceGRU_2/Shape:output:0"GRU_2/strided_slice/stack:output:0$GRU_2/strided_slice/stack_1:output:0$GRU_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_sliceh
GRU_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/mul/y
GRU_2/zeros/mulMulGRU_2/strided_slice:output:0GRU_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/mulk
GRU_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_2/zeros/Less/y
GRU_2/zeros/LessLessGRU_2/zeros/mul:z:0GRU_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/Lessn
GRU_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/packed/1
GRU_2/zeros/packedPackGRU_2/strided_slice:output:0GRU_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_2/zeros/packedk
GRU_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/zeros/Const
GRU_2/zerosFillGRU_2/zeros/packed:output:0GRU_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/zeros
GRU_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose/perm
GRU_2/transpose	TransposeGRU_1/transpose_1:y:0GRU_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transposea
GRU_2/Shape_1ShapeGRU_2/transpose:y:0*
T0*
_output_shapes
:2
GRU_2/Shape_1
GRU_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_1/stack
GRU_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_1
GRU_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_2
GRU_2/strided_slice_1StridedSliceGRU_2/Shape_1:output:0$GRU_2/strided_slice_1/stack:output:0&GRU_2/strided_slice_1/stack_1:output:0&GRU_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_slice_1
!GRU_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/TensorArrayV2/element_shapeЪ
GRU_2/TensorArrayV2TensorListReserve*GRU_2/TensorArrayV2/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2Ы
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_2/transpose:y:0DGRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_2/TensorArrayUnstack/TensorListFromTensor
GRU_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_2/stack
GRU_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_1
GRU_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_2 
GRU_2/strided_slice_2StridedSliceGRU_2/transpose:y:0$GRU_2/strided_slice_2/stack:output:0&GRU_2/strided_slice_2/stack_1:output:0&GRU_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_2Ў
 GRU_2/gru_cell_27/ReadVariableOpReadVariableOp)gru_2_gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_27/ReadVariableOp 
GRU_2/gru_cell_27/unstackUnpack(GRU_2/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_27/unstackУ
'GRU_2/gru_cell_27/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_27/MatMul/ReadVariableOpС
GRU_2/gru_cell_27/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMulЛ
GRU_2/gru_cell_27/BiasAddBiasAdd"GRU_2/gru_cell_27/MatMul:product:0"GRU_2/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAddt
GRU_2/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_27/Const
!GRU_2/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_27/split/split_dimє
GRU_2/gru_cell_27/splitSplit*GRU_2/gru_cell_27/split/split_dim:output:0"GRU_2/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/splitЩ
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_27/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMul_1С
GRU_2/gru_cell_27/BiasAdd_1BiasAdd$GRU_2/gru_cell_27/MatMul_1:product:0"GRU_2/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAdd_1
GRU_2/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_27/Const_1
#GRU_2/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_27/split_1/split_dim­
GRU_2/gru_cell_27/split_1SplitV$GRU_2/gru_cell_27/BiasAdd_1:output:0"GRU_2/gru_cell_27/Const_1:output:0,GRU_2/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/split_1Џ
GRU_2/gru_cell_27/addAddV2 GRU_2/gru_cell_27/split:output:0"GRU_2/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add
GRU_2/gru_cell_27/SigmoidSigmoidGRU_2/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/SigmoidГ
GRU_2/gru_cell_27/add_1AddV2 GRU_2/gru_cell_27/split:output:1"GRU_2/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_1
GRU_2/gru_cell_27/Sigmoid_1SigmoidGRU_2/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Sigmoid_1Ќ
GRU_2/gru_cell_27/mulMulGRU_2/gru_cell_27/Sigmoid_1:y:0"GRU_2/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mulЊ
GRU_2/gru_cell_27/add_2AddV2 GRU_2/gru_cell_27/split:output:2GRU_2/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_2
GRU_2/gru_cell_27/TanhTanhGRU_2/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Tanh 
GRU_2/gru_cell_27/mul_1MulGRU_2/gru_cell_27/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_1w
GRU_2/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_27/sub/xЈ
GRU_2/gru_cell_27/subSub GRU_2/gru_cell_27/sub/x:output:0GRU_2/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/subЂ
GRU_2/gru_cell_27/mul_2MulGRU_2/gru_cell_27/sub:z:0GRU_2/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_2Ї
GRU_2/gru_cell_27/add_3AddV2GRU_2/gru_cell_27/mul_1:z:0GRU_2/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_3
#GRU_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_2/TensorArrayV2_1/element_shapeа
GRU_2/TensorArrayV2_1TensorListReserve,GRU_2/TensorArrayV2_1/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2_1Z

GRU_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_2/time
GRU_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_2/while/maximum_iterationsv
GRU_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_2/while/loop_counterў
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_27_readvariableop_resource0gru_2_gru_cell_27_matmul_readvariableop_resource2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_2_while_body_694929*#
condR
GRU_2_while_cond_694928*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_2/whileС
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_2/TensorArrayV2Stack/TensorListStackTensorListStackGRU_2/while:output:3?GRU_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_2/TensorArrayV2Stack/TensorListStack
GRU_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_2/strided_slice_3/stack
GRU_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_3/stack_1
GRU_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_3/stack_2О
GRU_2/strided_slice_3StridedSlice1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_2/strided_slice_3/stack:output:0&GRU_2/strided_slice_3/stack_1:output:0&GRU_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_3
GRU_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose_1/permН
GRU_2/transpose_1	Transpose1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0GRU_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transpose_1r
GRU_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/runtimeЂ
OUT/Tensordot/ReadVariableOpReadVariableOp%out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
OUT/Tensordot/ReadVariableOpr
OUT/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/axesy
OUT/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
OUT/Tensordot/freeo
OUT/Tensordot/ShapeShapeGRU_2/transpose_1:y:0*
T0*
_output_shapes
:2
OUT/Tensordot/Shape|
OUT/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2/axisх
OUT/Tensordot/GatherV2GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/free:output:0$OUT/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2
OUT/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2_1/axisы
OUT/Tensordot/GatherV2_1GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/axes:output:0&OUT/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2_1t
OUT/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const
OUT/Tensordot/ProdProdOUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prodx
OUT/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const_1
OUT/Tensordot/Prod_1Prod!OUT/Tensordot/GatherV2_1:output:0OUT/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prod_1x
OUT/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat/axisФ
OUT/Tensordot/concatConcatV2OUT/Tensordot/free:output:0OUT/Tensordot/axes:output:0"OUT/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat
OUT/Tensordot/stackPackOUT/Tensordot/Prod:output:0OUT/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/stackЋ
OUT/Tensordot/transpose	TransposeGRU_2/transpose_1:y:0OUT/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/transposeЏ
OUT/Tensordot/ReshapeReshapeOUT/Tensordot/transpose:y:0OUT/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
OUT/Tensordot/ReshapeЎ
OUT/Tensordot/MatMulMatMulOUT/Tensordot/Reshape:output:0$OUT/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/MatMulx
OUT/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/Const_2|
OUT/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat_1/axisб
OUT/Tensordot/concat_1ConcatV2OUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const_2:output:0$OUT/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat_1 
OUT/TensordotReshapeOUT/Tensordot/MatMul:product:0OUT/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot
OUT/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
OUT/BiasAdd/ReadVariableOp
OUT/BiasAddBiasAddOUT/Tensordot:output:0"OUT/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/BiasAddq
OUT/SigmoidSigmoidOUT/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Sigmoid
IdentityIdentityOUT/Sigmoid:y:0^GRU_1/while^GRU_2/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::2
GRU_1/whileGRU_1/while2
GRU_2/whileGRU_2/while:X T
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameGRU_1_input
џ

GRU_1_while_cond_694773(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_694773___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_694773___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_694773___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_694773___redundant_placeholder3
gru_1_while_identity

GRU_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
GRU_1/while/Lesso
GRU_1/while/IdentityIdentityGRU_1/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_1/while/Identity"5
gru_1_while_identityGRU_1/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ЇX
ѕ
A__inference_GRU_2_layer_call_and_return_conditional_losses_696810
inputs_0'
#gru_cell_27_readvariableop_resource.
*gru_cell_27_matmul_readvariableop_resource0
,gru_cell_27_matmul_1_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_27/ReadVariableOp
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_27/unstackБ
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_27/MatMul/ReadVariableOpЉ
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMulЃ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAddh
gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_27/Const
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split/split_dimм
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/splitЗ
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_27/MatMul_1/ReadVariableOpЅ
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMul_1Љ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAdd_1
gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_27/Const_1
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split_1/split_dim
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const_1:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/split_1
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add|
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_1
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid_1
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_2u
gru_cell_27/TanhTanhgru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Tanh
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_1k
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_27/sub/x
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/sub
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_2
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_696720*
condR
while_cond_696719*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Б

&__inference_GRU_1_layer_call_fn_696141
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_1_layer_call_and_return_conditional_losses_6927942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
щ<
ж
A__inference_GRU_1_layer_call_and_return_conditional_losses_692794

inputs
gru_cell_26_692718
gru_cell_26_692720
gru_cell_26_692722
identityЂ#gru_cell_26/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2№
#gru_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_26_692718gru_cell_26_692720gru_cell_26_692722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_6924312%
#gru_cell_26/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_26_692718gru_cell_26_692720gru_cell_26_692722*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_692730*
condR
while_cond_692729*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0$^gru_cell_26/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_26/StatefulPartitionedCall#gru_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Б

&__inference_GRU_1_layer_call_fn_696152
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_1_layer_call_and_return_conditional_losses_6929122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
@
Е
while_body_697060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_27_readvariableop_resource_06
2while_gru_cell_27_matmul_readvariableop_resource_08
4while_gru_cell_27_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_27_readvariableop_resource4
0while_gru_cell_27_matmul_readvariableop_resource6
2while_gru_cell_27_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_27/ReadVariableOp 
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_27/unstackХ
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_27/MatMul/ReadVariableOpг
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMulЛ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAddt
while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_27/Const
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_27/split/split_dimє
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/splitЫ
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_27/MatMul_1/ReadVariableOpМ
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/MatMul_1С
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_27/BiasAdd_1
while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_27/Const_1
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_27/split_1/split_dim­
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0"while/gru_cell_27/Const_1:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_27/split_1Џ
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/SigmoidГ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_1
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Sigmoid_1Ќ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mulЊ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_2
while/gru_cell_27/TanhTanhwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/Tanh
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_1w
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_27/sub/xЈ
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/subЂ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/mul_2Ї
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_27/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
а
Њ
while_cond_695880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_695880___redundant_placeholder04
0while_while_cond_695880___redundant_placeholder14
0while_while_cond_695880___redundant_placeholder24
0while_while_cond_695880___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ц
щ
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_692993

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:<*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
шW
ѓ
A__inference_GRU_2_layer_call_and_return_conditional_losses_693992

inputs'
#gru_cell_27_readvariableop_resource.
*gru_cell_27_matmul_readvariableop_resource0
,gru_cell_27_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_27/ReadVariableOp
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_27/unstackБ
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_27/MatMul/ReadVariableOpЉ
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMulЃ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAddh
gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_27/Const
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split/split_dimм
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/splitЗ
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_27/MatMul_1/ReadVariableOpЅ
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMul_1Љ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAdd_1
gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_27/Const_1
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split_1/split_dim
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const_1:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/split_1
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add|
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_1
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid_1
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_2u
gru_cell_27/TanhTanhgru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Tanh
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_1k
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_27/sub/x
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/sub
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_2
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_693902*
condR
while_cond_693901*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъс

F__inference_Supervisor_layer_call_and_return_conditional_losses_695770

inputs-
)gru_1_gru_cell_26_readvariableop_resource4
0gru_1_gru_cell_26_matmul_readvariableop_resource6
2gru_1_gru_cell_26_matmul_1_readvariableop_resource-
)gru_2_gru_cell_27_readvariableop_resource4
0gru_2_gru_cell_27_matmul_readvariableop_resource6
2gru_2_gru_cell_27_matmul_1_readvariableop_resource)
%out_tensordot_readvariableop_resource'
#out_biasadd_readvariableop_resource
identityЂGRU_1/whileЂGRU_2/whileP
GRU_1/ShapeShapeinputs*
T0*
_output_shapes
:2
GRU_1/Shape
GRU_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice/stack
GRU_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_1
GRU_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice/stack_2
GRU_1/strided_sliceStridedSliceGRU_1/Shape:output:0"GRU_1/strided_slice/stack:output:0$GRU_1/strided_slice/stack_1:output:0$GRU_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_sliceh
GRU_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/mul/y
GRU_1/zeros/mulMulGRU_1/strided_slice:output:0GRU_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/mulk
GRU_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_1/zeros/Less/y
GRU_1/zeros/LessLessGRU_1/zeros/mul:z:0GRU_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_1/zeros/Lessn
GRU_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/zeros/packed/1
GRU_1/zeros/packedPackGRU_1/strided_slice:output:0GRU_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_1/zeros/packedk
GRU_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/zeros/Const
GRU_1/zerosFillGRU_1/zeros/packed:output:0GRU_1/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/zeros
GRU_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose/perm
GRU_1/transpose	TransposeinputsGRU_1/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transposea
GRU_1/Shape_1ShapeGRU_1/transpose:y:0*
T0*
_output_shapes
:2
GRU_1/Shape_1
GRU_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_1/stack
GRU_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_1
GRU_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_1/stack_2
GRU_1/strided_slice_1StridedSliceGRU_1/Shape_1:output:0$GRU_1/strided_slice_1/stack:output:0&GRU_1/strided_slice_1/stack_1:output:0&GRU_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_1/strided_slice_1
!GRU_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/TensorArrayV2/element_shapeЪ
GRU_1/TensorArrayV2TensorListReserve*GRU_1/TensorArrayV2/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2Ы
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_1/transpose:y:0DGRU_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_1/TensorArrayUnstack/TensorListFromTensor
GRU_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_2/stack
GRU_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_1
GRU_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_2/stack_2 
GRU_1/strided_slice_2StridedSliceGRU_1/transpose:y:0$GRU_1/strided_slice_2/stack:output:0&GRU_1/strided_slice_2/stack_1:output:0&GRU_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_2Ў
 GRU_1/gru_cell_26/ReadVariableOpReadVariableOp)gru_1_gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_26/ReadVariableOp 
GRU_1/gru_cell_26/unstackUnpack(GRU_1/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_26/unstackУ
'GRU_1/gru_cell_26/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_26/MatMul/ReadVariableOpС
GRU_1/gru_cell_26/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMulЛ
GRU_1/gru_cell_26/BiasAddBiasAdd"GRU_1/gru_cell_26/MatMul:product:0"GRU_1/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAddt
GRU_1/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_26/Const
!GRU_1/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_26/split/split_dimє
GRU_1/gru_cell_26/splitSplit*GRU_1/gru_cell_26/split/split_dim:output:0"GRU_1/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/splitЩ
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_26/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_26/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/MatMul_1С
GRU_1/gru_cell_26/BiasAdd_1BiasAdd$GRU_1/gru_cell_26/MatMul_1:product:0"GRU_1/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_26/BiasAdd_1
GRU_1/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_26/Const_1
#GRU_1/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_26/split_1/split_dim­
GRU_1/gru_cell_26/split_1SplitV$GRU_1/gru_cell_26/BiasAdd_1:output:0"GRU_1/gru_cell_26/Const_1:output:0,GRU_1/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_26/split_1Џ
GRU_1/gru_cell_26/addAddV2 GRU_1/gru_cell_26/split:output:0"GRU_1/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add
GRU_1/gru_cell_26/SigmoidSigmoidGRU_1/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/SigmoidГ
GRU_1/gru_cell_26/add_1AddV2 GRU_1/gru_cell_26/split:output:1"GRU_1/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_1
GRU_1/gru_cell_26/Sigmoid_1SigmoidGRU_1/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Sigmoid_1Ќ
GRU_1/gru_cell_26/mulMulGRU_1/gru_cell_26/Sigmoid_1:y:0"GRU_1/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mulЊ
GRU_1/gru_cell_26/add_2AddV2 GRU_1/gru_cell_26/split:output:2GRU_1/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_2
GRU_1/gru_cell_26/TanhTanhGRU_1/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/Tanh 
GRU_1/gru_cell_26/mul_1MulGRU_1/gru_cell_26/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_1w
GRU_1/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_26/sub/xЈ
GRU_1/gru_cell_26/subSub GRU_1/gru_cell_26/sub/x:output:0GRU_1/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/subЂ
GRU_1/gru_cell_26/mul_2MulGRU_1/gru_cell_26/sub:z:0GRU_1/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/mul_2Ї
GRU_1/gru_cell_26/add_3AddV2GRU_1/gru_cell_26/mul_1:z:0GRU_1/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_26/add_3
#GRU_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_1/TensorArrayV2_1/element_shapeа
GRU_1/TensorArrayV2_1TensorListReserve,GRU_1/TensorArrayV2_1/element_shape:output:0GRU_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_1/TensorArrayV2_1Z

GRU_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_1/time
GRU_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_1/while/maximum_iterationsv
GRU_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_1/while/loop_counterў
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_26_readvariableop_resource0gru_1_gru_cell_26_matmul_readvariableop_resource2gru_1_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_1_while_body_695498*#
condR
GRU_1_while_cond_695497*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_1/whileС
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_1/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_1/TensorArrayV2Stack/TensorListStackTensorListStackGRU_1/while:output:3?GRU_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_1/TensorArrayV2Stack/TensorListStack
GRU_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_1/strided_slice_3/stack
GRU_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_1/strided_slice_3/stack_1
GRU_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_1/strided_slice_3/stack_2О
GRU_1/strided_slice_3StridedSlice1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_1/strided_slice_3/stack:output:0&GRU_1/strided_slice_3/stack_1:output:0&GRU_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_1/strided_slice_3
GRU_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_1/transpose_1/permН
GRU_1/transpose_1	Transpose1GRU_1/TensorArrayV2Stack/TensorListStack:tensor:0GRU_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_1/transpose_1r
GRU_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_1/runtime_
GRU_2/ShapeShapeGRU_1/transpose_1:y:0*
T0*
_output_shapes
:2
GRU_2/Shape
GRU_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice/stack
GRU_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_1
GRU_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice/stack_2
GRU_2/strided_sliceStridedSliceGRU_2/Shape:output:0"GRU_2/strided_slice/stack:output:0$GRU_2/strided_slice/stack_1:output:0$GRU_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_sliceh
GRU_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/mul/y
GRU_2/zeros/mulMulGRU_2/strided_slice:output:0GRU_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/mulk
GRU_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
GRU_2/zeros/Less/y
GRU_2/zeros/LessLessGRU_2/zeros/mul:z:0GRU_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
GRU_2/zeros/Lessn
GRU_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/zeros/packed/1
GRU_2/zeros/packedPackGRU_2/strided_slice:output:0GRU_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
GRU_2/zeros/packedk
GRU_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/zeros/Const
GRU_2/zerosFillGRU_2/zeros/packed:output:0GRU_2/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/zeros
GRU_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose/perm
GRU_2/transpose	TransposeGRU_1/transpose_1:y:0GRU_2/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transposea
GRU_2/Shape_1ShapeGRU_2/transpose:y:0*
T0*
_output_shapes
:2
GRU_2/Shape_1
GRU_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_1/stack
GRU_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_1
GRU_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_1/stack_2
GRU_2/strided_slice_1StridedSliceGRU_2/Shape_1:output:0$GRU_2/strided_slice_1/stack:output:0&GRU_2/strided_slice_1/stack_1:output:0&GRU_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
GRU_2/strided_slice_1
!GRU_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/TensorArrayV2/element_shapeЪ
GRU_2/TensorArrayV2TensorListReserve*GRU_2/TensorArrayV2/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2Ы
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;GRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape
-GRU_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorGRU_2/transpose:y:0DGRU_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-GRU_2/TensorArrayUnstack/TensorListFromTensor
GRU_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_2/stack
GRU_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_1
GRU_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_2/stack_2 
GRU_2/strided_slice_2StridedSliceGRU_2/transpose:y:0$GRU_2/strided_slice_2/stack:output:0&GRU_2/strided_slice_2/stack_1:output:0&GRU_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_2Ў
 GRU_2/gru_cell_27/ReadVariableOpReadVariableOp)gru_2_gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_27/ReadVariableOp 
GRU_2/gru_cell_27/unstackUnpack(GRU_2/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_27/unstackУ
'GRU_2/gru_cell_27/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_27/MatMul/ReadVariableOpС
GRU_2/gru_cell_27/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMulЛ
GRU_2/gru_cell_27/BiasAddBiasAdd"GRU_2/gru_cell_27/MatMul:product:0"GRU_2/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAddt
GRU_2/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_27/Const
!GRU_2/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_27/split/split_dimє
GRU_2/gru_cell_27/splitSplit*GRU_2/gru_cell_27/split/split_dim:output:0"GRU_2/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/splitЩ
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_27/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_27/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/MatMul_1С
GRU_2/gru_cell_27/BiasAdd_1BiasAdd$GRU_2/gru_cell_27/MatMul_1:product:0"GRU_2/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_27/BiasAdd_1
GRU_2/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_27/Const_1
#GRU_2/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_27/split_1/split_dim­
GRU_2/gru_cell_27/split_1SplitV$GRU_2/gru_cell_27/BiasAdd_1:output:0"GRU_2/gru_cell_27/Const_1:output:0,GRU_2/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_27/split_1Џ
GRU_2/gru_cell_27/addAddV2 GRU_2/gru_cell_27/split:output:0"GRU_2/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add
GRU_2/gru_cell_27/SigmoidSigmoidGRU_2/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/SigmoidГ
GRU_2/gru_cell_27/add_1AddV2 GRU_2/gru_cell_27/split:output:1"GRU_2/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_1
GRU_2/gru_cell_27/Sigmoid_1SigmoidGRU_2/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Sigmoid_1Ќ
GRU_2/gru_cell_27/mulMulGRU_2/gru_cell_27/Sigmoid_1:y:0"GRU_2/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mulЊ
GRU_2/gru_cell_27/add_2AddV2 GRU_2/gru_cell_27/split:output:2GRU_2/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_2
GRU_2/gru_cell_27/TanhTanhGRU_2/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/Tanh 
GRU_2/gru_cell_27/mul_1MulGRU_2/gru_cell_27/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_1w
GRU_2/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_27/sub/xЈ
GRU_2/gru_cell_27/subSub GRU_2/gru_cell_27/sub/x:output:0GRU_2/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/subЂ
GRU_2/gru_cell_27/mul_2MulGRU_2/gru_cell_27/sub:z:0GRU_2/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/mul_2Ї
GRU_2/gru_cell_27/add_3AddV2GRU_2/gru_cell_27/mul_1:z:0GRU_2/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_27/add_3
#GRU_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2%
#GRU_2/TensorArrayV2_1/element_shapeа
GRU_2/TensorArrayV2_1TensorListReserve,GRU_2/TensorArrayV2_1/element_shape:output:0GRU_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
GRU_2/TensorArrayV2_1Z

GRU_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

GRU_2/time
GRU_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
GRU_2/while/maximum_iterationsv
GRU_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
GRU_2/while/loop_counterў
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_27_readvariableop_resource0gru_2_gru_cell_27_matmul_readvariableop_resource2gru_2_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*#
bodyR
GRU_2_while_body_695653*#
condR
GRU_2_while_cond_695652*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
GRU_2/whileС
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   28
6GRU_2/TensorArrayV2Stack/TensorListStack/element_shape
(GRU_2/TensorArrayV2Stack/TensorListStackTensorListStackGRU_2/while:output:3?GRU_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02*
(GRU_2/TensorArrayV2Stack/TensorListStack
GRU_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
GRU_2/strided_slice_3/stack
GRU_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
GRU_2/strided_slice_3/stack_1
GRU_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
GRU_2/strided_slice_3/stack_2О
GRU_2/strided_slice_3StridedSlice1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0$GRU_2/strided_slice_3/stack:output:0&GRU_2/strided_slice_3/stack_1:output:0&GRU_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
GRU_2/strided_slice_3
GRU_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
GRU_2/transpose_1/permН
GRU_2/transpose_1	Transpose1GRU_2/TensorArrayV2Stack/TensorListStack:tensor:0GRU_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
GRU_2/transpose_1r
GRU_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
GRU_2/runtimeЂ
OUT/Tensordot/ReadVariableOpReadVariableOp%out_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
OUT/Tensordot/ReadVariableOpr
OUT/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/axesy
OUT/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
OUT/Tensordot/freeo
OUT/Tensordot/ShapeShapeGRU_2/transpose_1:y:0*
T0*
_output_shapes
:2
OUT/Tensordot/Shape|
OUT/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2/axisх
OUT/Tensordot/GatherV2GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/free:output:0$OUT/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2
OUT/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/GatherV2_1/axisы
OUT/Tensordot/GatherV2_1GatherV2OUT/Tensordot/Shape:output:0OUT/Tensordot/axes:output:0&OUT/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
OUT/Tensordot/GatherV2_1t
OUT/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const
OUT/Tensordot/ProdProdOUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prodx
OUT/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
OUT/Tensordot/Const_1
OUT/Tensordot/Prod_1Prod!OUT/Tensordot/GatherV2_1:output:0OUT/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
OUT/Tensordot/Prod_1x
OUT/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat/axisФ
OUT/Tensordot/concatConcatV2OUT/Tensordot/free:output:0OUT/Tensordot/axes:output:0"OUT/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat
OUT/Tensordot/stackPackOUT/Tensordot/Prod:output:0OUT/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/stackЋ
OUT/Tensordot/transpose	TransposeGRU_2/transpose_1:y:0OUT/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/transposeЏ
OUT/Tensordot/ReshapeReshapeOUT/Tensordot/transpose:y:0OUT/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
OUT/Tensordot/ReshapeЎ
OUT/Tensordot/MatMulMatMulOUT/Tensordot/Reshape:output:0$OUT/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot/MatMulx
OUT/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
OUT/Tensordot/Const_2|
OUT/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
OUT/Tensordot/concat_1/axisб
OUT/Tensordot/concat_1ConcatV2OUT/Tensordot/GatherV2:output:0OUT/Tensordot/Const_2:output:0$OUT/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
OUT/Tensordot/concat_1 
OUT/TensordotReshapeOUT/Tensordot/MatMul:product:0OUT/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Tensordot
OUT/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
OUT/BiasAdd/ReadVariableOp
OUT/BiasAddBiasAddOUT/Tensordot:output:0"OUT/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/BiasAddq
OUT/SigmoidSigmoidOUT/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
OUT/Sigmoid
IdentityIdentityOUT/Sigmoid:y:0^GRU_1/while^GRU_2/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::2
GRU_1/whileGRU_1/while2
GRU_2/whileGRU_2/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ<
ж
A__inference_GRU_2_layer_call_and_return_conditional_losses_693356

inputs
gru_cell_27_693280
gru_cell_27_693282
gru_cell_27_693284
identityЂ#gru_cell_27/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2№
#gru_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_27_693280gru_cell_27_693282gru_cell_27_693284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_6929932%
#gru_cell_27/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterч
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_27_693280gru_cell_27_693282gru_cell_27_693284*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_693292*
condR
while_cond_693291*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0$^gru_cell_27/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_27/StatefulPartitionedCall#gru_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
Њ
while_cond_696039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_696039___redundant_placeholder04
0while_while_cond_696039___redundant_placeholder14
0while_while_cond_696039___redundant_placeholder24
0while_while_cond_696039___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
а
Њ
while_cond_696900
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_696900___redundant_placeholder04
0while_while_cond_696900___redundant_placeholder14
0while_while_cond_696900___redundant_placeholder24
0while_while_cond_696900___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
а
Њ
while_cond_692729
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_692729___redundant_placeholder04
0while_while_cond_692729___redundant_placeholder14
0while_while_cond_692729___redundant_placeholder24
0while_while_cond_692729___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

и
$__inference_signature_wrapper_694364
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_6923592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameGRU_1_input
а
Њ
while_cond_693291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_693291___redundant_placeholder04
0while_while_cond_693291___redundant_placeholder14
0while_while_cond_693291___redundant_placeholder24
0while_while_cond_693291___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
нH
з
GRU_1_while_body_695498(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_26_readvariableop_resource_0<
8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_26_readvariableop_resource:
6gru_1_while_gru_cell_26_matmul_readvariableop_resource<
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resourceЯ
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFGRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_1/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_1/while/gru_cell_26/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_26/ReadVariableOpВ
GRU_1/while/gru_cell_26/unstackUnpack.GRU_1/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_26/unstackз
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_26/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_26/MatMulг
GRU_1/while/gru_cell_26/BiasAddBiasAdd(GRU_1/while/gru_cell_26/MatMul:product:0(GRU_1/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_26/BiasAdd
GRU_1/while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_26/Const
'GRU_1/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_26/split/split_dim
GRU_1/while/gru_cell_26/splitSplit0GRU_1/while/gru_cell_26/split/split_dim:output:0(GRU_1/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_26/splitн
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_26/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_26/MatMul_1й
!GRU_1/while/gru_cell_26/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_26/MatMul_1:product:0(GRU_1/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_26/BiasAdd_1
GRU_1/while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_26/Const_1Ё
)GRU_1/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_26/split_1/split_dimЫ
GRU_1/while/gru_cell_26/split_1SplitV*GRU_1/while/gru_cell_26/BiasAdd_1:output:0(GRU_1/while/gru_cell_26/Const_1:output:02GRU_1/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_26/split_1Ч
GRU_1/while/gru_cell_26/addAddV2&GRU_1/while/gru_cell_26/split:output:0(GRU_1/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add 
GRU_1/while/gru_cell_26/SigmoidSigmoidGRU_1/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_26/SigmoidЫ
GRU_1/while/gru_cell_26/add_1AddV2&GRU_1/while/gru_cell_26/split:output:1(GRU_1/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_1І
!GRU_1/while/gru_cell_26/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_26/Sigmoid_1Ф
GRU_1/while/gru_cell_26/mulMul%GRU_1/while/gru_cell_26/Sigmoid_1:y:0(GRU_1/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mulТ
GRU_1/while/gru_cell_26/add_2AddV2&GRU_1/while/gru_cell_26/split:output:2GRU_1/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_2
GRU_1/while/gru_cell_26/TanhTanh!GRU_1/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/TanhЗ
GRU_1/while/gru_cell_26/mul_1Mul#GRU_1/while/gru_cell_26/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_1
GRU_1/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_26/sub/xР
GRU_1/while/gru_cell_26/subSub&GRU_1/while/gru_cell_26/sub/x:output:0#GRU_1/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/subК
GRU_1/while/gru_cell_26/mul_2MulGRU_1/while/gru_cell_26/sub:z:0 GRU_1/while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_2П
GRU_1/while/gru_cell_26/add_3AddV2!GRU_1/while/gru_cell_26/mul_1:z:0!GRU_1/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_1/while/TensorArrayV2Write/TensorListSetItemh
GRU_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add/y
GRU_1/while/addAddV2gru_1_while_placeholderGRU_1/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/addl
GRU_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add_1/y
GRU_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_counterGRU_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/add_1p
GRU_1/while/IdentityIdentityGRU_1/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity
GRU_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_1/while/Identity_1r
GRU_1/while/Identity_2IdentityGRU_1/while/add:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_2
GRU_1/while/Identity_3Identity@GRU_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_3
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resource:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_26_matmul_readvariableop_resource8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_26_readvariableop_resource1gru_1_while_gru_cell_26_readvariableop_resource_0"5
gru_1_while_identityGRU_1/while/Identity:output:0"9
gru_1_while_identity_1GRU_1/while/Identity_1:output:0"9
gru_1_while_identity_2GRU_1/while/Identity_2:output:0"9
gru_1_while_identity_3GRU_1/while/Identity_3:output:0"9
gru_1_while_identity_4GRU_1/while/Identity_4:output:0"Р
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
нH
з
GRU_2_while_body_695312(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_27_readvariableop_resource_0<
8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_27_readvariableop_resource:
6gru_2_while_gru_cell_27_matmul_readvariableop_resource<
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resourceЯ
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0gru_2_while_placeholderFGRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_2/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_2/while/gru_cell_27/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_27/ReadVariableOpВ
GRU_2/while/gru_cell_27/unstackUnpack.GRU_2/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_27/unstackз
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_27/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_27/MatMulг
GRU_2/while/gru_cell_27/BiasAddBiasAdd(GRU_2/while/gru_cell_27/MatMul:product:0(GRU_2/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_27/BiasAdd
GRU_2/while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_27/Const
'GRU_2/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_27/split/split_dim
GRU_2/while/gru_cell_27/splitSplit0GRU_2/while/gru_cell_27/split/split_dim:output:0(GRU_2/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_27/splitн
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_27/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_27/MatMul_1й
!GRU_2/while/gru_cell_27/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_27/MatMul_1:product:0(GRU_2/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_27/BiasAdd_1
GRU_2/while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_27/Const_1Ё
)GRU_2/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_27/split_1/split_dimЫ
GRU_2/while/gru_cell_27/split_1SplitV*GRU_2/while/gru_cell_27/BiasAdd_1:output:0(GRU_2/while/gru_cell_27/Const_1:output:02GRU_2/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_27/split_1Ч
GRU_2/while/gru_cell_27/addAddV2&GRU_2/while/gru_cell_27/split:output:0(GRU_2/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add 
GRU_2/while/gru_cell_27/SigmoidSigmoidGRU_2/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_27/SigmoidЫ
GRU_2/while/gru_cell_27/add_1AddV2&GRU_2/while/gru_cell_27/split:output:1(GRU_2/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_1І
!GRU_2/while/gru_cell_27/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_27/Sigmoid_1Ф
GRU_2/while/gru_cell_27/mulMul%GRU_2/while/gru_cell_27/Sigmoid_1:y:0(GRU_2/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mulТ
GRU_2/while/gru_cell_27/add_2AddV2&GRU_2/while/gru_cell_27/split:output:2GRU_2/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_2
GRU_2/while/gru_cell_27/TanhTanh!GRU_2/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/TanhЗ
GRU_2/while/gru_cell_27/mul_1Mul#GRU_2/while/gru_cell_27/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_1
GRU_2/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_27/sub/xР
GRU_2/while/gru_cell_27/subSub&GRU_2/while/gru_cell_27/sub/x:output:0#GRU_2/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/subК
GRU_2/while/gru_cell_27/mul_2MulGRU_2/while/gru_cell_27/sub:z:0 GRU_2/while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_2П
GRU_2/while/gru_cell_27/add_3AddV2!GRU_2/while/gru_cell_27/mul_1:z:0!GRU_2/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_2/while/TensorArrayV2Write/TensorListSetItemh
GRU_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add/y
GRU_2/while/addAddV2gru_2_while_placeholderGRU_2/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/addl
GRU_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add_1/y
GRU_2/while/add_1AddV2$gru_2_while_gru_2_while_loop_counterGRU_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/add_1p
GRU_2/while/IdentityIdentityGRU_2/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity
GRU_2/while/Identity_1Identity*gru_2_while_gru_2_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_2/while/Identity_1r
GRU_2/while/Identity_2IdentityGRU_2/while/add:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_2
GRU_2/while/Identity_3Identity@GRU_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_3
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resource:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_27_matmul_readvariableop_resource8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_27_readvariableop_resource1gru_2_while_gru_cell_27_readvariableop_resource_0"5
gru_2_while_identityGRU_2/while/Identity:output:0"9
gru_2_while_identity_1GRU_2/while/Identity_1:output:0"9
gru_2_while_identity_2GRU_2/while/Identity_2:output:0"9
gru_2_while_identity_3GRU_2/while/Identity_3:output:0"9
gru_2_while_identity_4GRU_2/while/Identity_4:output:0"Р
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ѕ
Я
F__inference_Supervisor_layer_call_and_return_conditional_losses_694322

inputs
gru_1_694302
gru_1_694304
gru_1_694306
gru_2_694309
gru_2_694311
gru_2_694313

out_694316

out_694318
identityЂGRU_1/StatefulPartitionedCallЂGRU_2/StatefulPartitionedCallЂOUT/StatefulPartitionedCall
GRU_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_694302gru_1_694304gru_1_694306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_1_layer_call_and_return_conditional_losses_6938042
GRU_1/StatefulPartitionedCallЙ
GRU_2/StatefulPartitionedCallStatefulPartitionedCall&GRU_1/StatefulPartitionedCall:output:0gru_2_694309gru_2_694311gru_2_694313*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GRU_2_layer_call_and_return_conditional_losses_6941512
GRU_2/StatefulPartitionedCall
OUT/StatefulPartitionedCallStatefulPartitionedCall&GRU_2/StatefulPartitionedCall:output:0
out_694316
out_694318*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_OUT_layer_call_and_return_conditional_losses_6942122
OUT/StatefulPartitionedCallк
IdentityIdentity$OUT/StatefulPartitionedCall:output:0^GRU_1/StatefulPartitionedCall^GRU_2/StatefulPartitionedCall^OUT/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ::::::::2>
GRU_1/StatefulPartitionedCallGRU_1/StatefulPartitionedCall2>
GRU_2/StatefulPartitionedCallGRU_2/StatefulPartitionedCall2:
OUT/StatefulPartitionedCallOUT/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

GRU_2_while_cond_695311(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1@
<gru_2_while_gru_2_while_cond_695311___redundant_placeholder0@
<gru_2_while_gru_2_while_cond_695311___redundant_placeholder1@
<gru_2_while_gru_2_while_cond_695311___redundant_placeholder2@
<gru_2_while_gru_2_while_cond_695311___redundant_placeholder3
gru_2_while_identity

GRU_2/while/LessLessgru_2_while_placeholder&gru_2_while_less_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
GRU_2/while/Lesso
GRU_2/while/IdentityIdentityGRU_2/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_2/while/Identity"5
gru_2_while_identityGRU_2/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
џ

GRU_1_while_cond_695156(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_695156___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_695156___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_695156___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_695156___redundant_placeholder3
gru_1_while_identity

GRU_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
GRU_1/while/Lesso
GRU_1/while/IdentityIdentityGRU_1/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_1/while/Identity"5
gru_1_while_identityGRU_1/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
нH
з
GRU_2_while_body_694929(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_27_readvariableop_resource_0<
8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_27_readvariableop_resource:
6gru_2_while_gru_cell_27_matmul_readvariableop_resource<
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resourceЯ
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0gru_2_while_placeholderFGRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_2/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_2/while/gru_cell_27/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_27/ReadVariableOpВ
GRU_2/while/gru_cell_27/unstackUnpack.GRU_2/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_27/unstackз
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_27/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_27/MatMulг
GRU_2/while/gru_cell_27/BiasAddBiasAdd(GRU_2/while/gru_cell_27/MatMul:product:0(GRU_2/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_27/BiasAdd
GRU_2/while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_27/Const
'GRU_2/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_27/split/split_dim
GRU_2/while/gru_cell_27/splitSplit0GRU_2/while/gru_cell_27/split/split_dim:output:0(GRU_2/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_27/splitн
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_27/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_27/MatMul_1й
!GRU_2/while/gru_cell_27/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_27/MatMul_1:product:0(GRU_2/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_27/BiasAdd_1
GRU_2/while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_27/Const_1Ё
)GRU_2/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_27/split_1/split_dimЫ
GRU_2/while/gru_cell_27/split_1SplitV*GRU_2/while/gru_cell_27/BiasAdd_1:output:0(GRU_2/while/gru_cell_27/Const_1:output:02GRU_2/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_27/split_1Ч
GRU_2/while/gru_cell_27/addAddV2&GRU_2/while/gru_cell_27/split:output:0(GRU_2/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add 
GRU_2/while/gru_cell_27/SigmoidSigmoidGRU_2/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_27/SigmoidЫ
GRU_2/while/gru_cell_27/add_1AddV2&GRU_2/while/gru_cell_27/split:output:1(GRU_2/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_1І
!GRU_2/while/gru_cell_27/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_27/Sigmoid_1Ф
GRU_2/while/gru_cell_27/mulMul%GRU_2/while/gru_cell_27/Sigmoid_1:y:0(GRU_2/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mulТ
GRU_2/while/gru_cell_27/add_2AddV2&GRU_2/while/gru_cell_27/split:output:2GRU_2/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_2
GRU_2/while/gru_cell_27/TanhTanh!GRU_2/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/TanhЗ
GRU_2/while/gru_cell_27/mul_1Mul#GRU_2/while/gru_cell_27/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_1
GRU_2/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_27/sub/xР
GRU_2/while/gru_cell_27/subSub&GRU_2/while/gru_cell_27/sub/x:output:0#GRU_2/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/subК
GRU_2/while/gru_cell_27/mul_2MulGRU_2/while/gru_cell_27/sub:z:0 GRU_2/while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_2П
GRU_2/while/gru_cell_27/add_3AddV2!GRU_2/while/gru_cell_27/mul_1:z:0!GRU_2/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_2/while/TensorArrayV2Write/TensorListSetItemh
GRU_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add/y
GRU_2/while/addAddV2gru_2_while_placeholderGRU_2/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/addl
GRU_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add_1/y
GRU_2/while/add_1AddV2$gru_2_while_gru_2_while_loop_counterGRU_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/add_1p
GRU_2/while/IdentityIdentityGRU_2/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity
GRU_2/while/Identity_1Identity*gru_2_while_gru_2_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_2/while/Identity_1r
GRU_2/while/Identity_2IdentityGRU_2/while/add:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_2
GRU_2/while/Identity_3Identity@GRU_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_3
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resource:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_27_matmul_readvariableop_resource8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_27_readvariableop_resource1gru_2_while_gru_cell_27_readvariableop_resource_0"5
gru_2_while_identityGRU_2/while/Identity:output:0"9
gru_2_while_identity_1GRU_2/while/Identity_1:output:0"9
gru_2_while_identity_2GRU_2/while/Identity_2:output:0"9
gru_2_while_identity_3GRU_2/while/Identity_3:output:0"9
gru_2_while_identity_4GRU_2/while/Identity_4:output:0"Р
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
џ

GRU_2_while_cond_695652(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1@
<gru_2_while_gru_2_while_cond_695652___redundant_placeholder0@
<gru_2_while_gru_2_while_cond_695652___redundant_placeholder1@
<gru_2_while_gru_2_while_cond_695652___redundant_placeholder2@
<gru_2_while_gru_2_while_cond_695652___redundant_placeholder3
gru_2_while_identity

GRU_2/while/LessLessgru_2_while_placeholder&gru_2_while_less_gru_2_strided_slice_1*
T0*
_output_shapes
: 2
GRU_2/while/Lesso
GRU_2/while/IdentityIdentityGRU_2/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_2/while/Identity"5
gru_2_while_identityGRU_2/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ЉX


"Supervisor_GRU_1_while_body_692087>
:supervisor_gru_1_while_supervisor_gru_1_while_loop_counterD
@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations&
"supervisor_gru_1_while_placeholder(
$supervisor_gru_1_while_placeholder_1(
$supervisor_gru_1_while_placeholder_2=
9supervisor_gru_1_while_supervisor_gru_1_strided_slice_1_0y
usupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor_0@
<supervisor_gru_1_while_gru_cell_26_readvariableop_resource_0G
Csupervisor_gru_1_while_gru_cell_26_matmul_readvariableop_resource_0I
Esupervisor_gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0#
supervisor_gru_1_while_identity%
!supervisor_gru_1_while_identity_1%
!supervisor_gru_1_while_identity_2%
!supervisor_gru_1_while_identity_3%
!supervisor_gru_1_while_identity_4;
7supervisor_gru_1_while_supervisor_gru_1_strided_slice_1w
ssupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor>
:supervisor_gru_1_while_gru_cell_26_readvariableop_resourceE
Asupervisor_gru_1_while_gru_cell_26_matmul_readvariableop_resourceG
Csupervisor_gru_1_while_gru_cell_26_matmul_1_readvariableop_resourceх
HSupervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2J
HSupervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeЙ
:Supervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor_0"supervisor_gru_1_while_placeholderQSupervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02<
:Supervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItemу
1Supervisor/GRU_1/while/gru_cell_26/ReadVariableOpReadVariableOp<supervisor_gru_1_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype023
1Supervisor/GRU_1/while/gru_cell_26/ReadVariableOpг
*Supervisor/GRU_1/while/gru_cell_26/unstackUnpack9Supervisor/GRU_1/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2,
*Supervisor/GRU_1/while/gru_cell_26/unstackј
8Supervisor/GRU_1/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOpCsupervisor_gru_1_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02:
8Supervisor/GRU_1/while/gru_cell_26/MatMul/ReadVariableOp
)Supervisor/GRU_1/while/gru_cell_26/MatMulMatMulASupervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:0@Supervisor/GRU_1/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2+
)Supervisor/GRU_1/while/gru_cell_26/MatMulџ
*Supervisor/GRU_1/while/gru_cell_26/BiasAddBiasAdd3Supervisor/GRU_1/while/gru_cell_26/MatMul:product:03Supervisor/GRU_1/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2,
*Supervisor/GRU_1/while/gru_cell_26/BiasAdd
(Supervisor/GRU_1/while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(Supervisor/GRU_1/while/gru_cell_26/ConstГ
2Supervisor/GRU_1/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2Supervisor/GRU_1/while/gru_cell_26/split/split_dimИ
(Supervisor/GRU_1/while/gru_cell_26/splitSplit;Supervisor/GRU_1/while/gru_cell_26/split/split_dim:output:03Supervisor/GRU_1/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2*
(Supervisor/GRU_1/while/gru_cell_26/splitў
:Supervisor/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOpEsupervisor_gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02<
:Supervisor/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOp
+Supervisor/GRU_1/while/gru_cell_26/MatMul_1MatMul$supervisor_gru_1_while_placeholder_2BSupervisor/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2-
+Supervisor/GRU_1/while/gru_cell_26/MatMul_1
,Supervisor/GRU_1/while/gru_cell_26/BiasAdd_1BiasAdd5Supervisor/GRU_1/while/gru_cell_26/MatMul_1:product:03Supervisor/GRU_1/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2.
,Supervisor/GRU_1/while/gru_cell_26/BiasAdd_1­
*Supervisor/GRU_1/while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2,
*Supervisor/GRU_1/while/gru_cell_26/Const_1З
4Supervisor/GRU_1/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ26
4Supervisor/GRU_1/while/gru_cell_26/split_1/split_dim
*Supervisor/GRU_1/while/gru_cell_26/split_1SplitV5Supervisor/GRU_1/while/gru_cell_26/BiasAdd_1:output:03Supervisor/GRU_1/while/gru_cell_26/Const_1:output:0=Supervisor/GRU_1/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*Supervisor/GRU_1/while/gru_cell_26/split_1ѓ
&Supervisor/GRU_1/while/gru_cell_26/addAddV21Supervisor/GRU_1/while/gru_cell_26/split:output:03Supervisor/GRU_1/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_26/addС
*Supervisor/GRU_1/while/gru_cell_26/SigmoidSigmoid*Supervisor/GRU_1/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*Supervisor/GRU_1/while/gru_cell_26/Sigmoidї
(Supervisor/GRU_1/while/gru_cell_26/add_1AddV21Supervisor/GRU_1/while/gru_cell_26/split:output:13Supervisor/GRU_1/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_26/add_1Ч
,Supervisor/GRU_1/while/gru_cell_26/Sigmoid_1Sigmoid,Supervisor/GRU_1/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,Supervisor/GRU_1/while/gru_cell_26/Sigmoid_1№
&Supervisor/GRU_1/while/gru_cell_26/mulMul0Supervisor/GRU_1/while/gru_cell_26/Sigmoid_1:y:03Supervisor/GRU_1/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_26/mulю
(Supervisor/GRU_1/while/gru_cell_26/add_2AddV21Supervisor/GRU_1/while/gru_cell_26/split:output:2*Supervisor/GRU_1/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_26/add_2К
'Supervisor/GRU_1/while/gru_cell_26/TanhTanh,Supervisor/GRU_1/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'Supervisor/GRU_1/while/gru_cell_26/Tanhу
(Supervisor/GRU_1/while/gru_cell_26/mul_1Mul.Supervisor/GRU_1/while/gru_cell_26/Sigmoid:y:0$supervisor_gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_26/mul_1
(Supervisor/GRU_1/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(Supervisor/GRU_1/while/gru_cell_26/sub/xь
&Supervisor/GRU_1/while/gru_cell_26/subSub1Supervisor/GRU_1/while/gru_cell_26/sub/x:output:0.Supervisor/GRU_1/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_26/subц
(Supervisor/GRU_1/while/gru_cell_26/mul_2Mul*Supervisor/GRU_1/while/gru_cell_26/sub:z:0+Supervisor/GRU_1/while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_26/mul_2ы
(Supervisor/GRU_1/while/gru_cell_26/add_3AddV2,Supervisor/GRU_1/while/gru_cell_26/mul_1:z:0,Supervisor/GRU_1/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_26/add_3Д
;Supervisor/GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$supervisor_gru_1_while_placeholder_1"supervisor_gru_1_while_placeholder,Supervisor/GRU_1/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02=
;Supervisor/GRU_1/while/TensorArrayV2Write/TensorListSetItem~
Supervisor/GRU_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
Supervisor/GRU_1/while/add/y­
Supervisor/GRU_1/while/addAddV2"supervisor_gru_1_while_placeholder%Supervisor/GRU_1/while/add/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_1/while/add
Supervisor/GRU_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
Supervisor/GRU_1/while/add_1/yЫ
Supervisor/GRU_1/while/add_1AddV2:supervisor_gru_1_while_supervisor_gru_1_while_loop_counter'Supervisor/GRU_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
Supervisor/GRU_1/while/add_1
Supervisor/GRU_1/while/IdentityIdentity Supervisor/GRU_1/while/add_1:z:0*
T0*
_output_shapes
: 2!
Supervisor/GRU_1/while/IdentityЕ
!Supervisor/GRU_1/while/Identity_1Identity@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations*
T0*
_output_shapes
: 2#
!Supervisor/GRU_1/while/Identity_1
!Supervisor/GRU_1/while/Identity_2IdentitySupervisor/GRU_1/while/add:z:0*
T0*
_output_shapes
: 2#
!Supervisor/GRU_1/while/Identity_2Р
!Supervisor/GRU_1/while/Identity_3IdentityKSupervisor/GRU_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2#
!Supervisor/GRU_1/while/Identity_3В
!Supervisor/GRU_1/while/Identity_4Identity,Supervisor/GRU_1/while/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_1/while/Identity_4"
Csupervisor_gru_1_while_gru_cell_26_matmul_1_readvariableop_resourceEsupervisor_gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0"
Asupervisor_gru_1_while_gru_cell_26_matmul_readvariableop_resourceCsupervisor_gru_1_while_gru_cell_26_matmul_readvariableop_resource_0"z
:supervisor_gru_1_while_gru_cell_26_readvariableop_resource<supervisor_gru_1_while_gru_cell_26_readvariableop_resource_0"K
supervisor_gru_1_while_identity(Supervisor/GRU_1/while/Identity:output:0"O
!supervisor_gru_1_while_identity_1*Supervisor/GRU_1/while/Identity_1:output:0"O
!supervisor_gru_1_while_identity_2*Supervisor/GRU_1/while/Identity_2:output:0"O
!supervisor_gru_1_while_identity_3*Supervisor/GRU_1/while/Identity_3:output:0"O
!supervisor_gru_1_while_identity_4*Supervisor/GRU_1/while/Identity_4:output:0"t
7supervisor_gru_1_while_supervisor_gru_1_strided_slice_19supervisor_gru_1_while_supervisor_gru_1_strided_slice_1_0"ь
ssupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensorusupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
нH
з
GRU_2_while_body_695653(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_27_readvariableop_resource_0<
8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_27_readvariableop_resource:
6gru_2_while_gru_cell_27_matmul_readvariableop_resource<
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resourceЯ
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0gru_2_while_placeholderFGRU_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_2/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_2/while/gru_cell_27/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_27/ReadVariableOpВ
GRU_2/while/gru_cell_27/unstackUnpack.GRU_2/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_27/unstackз
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_27/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_27/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_27/MatMulг
GRU_2/while/gru_cell_27/BiasAddBiasAdd(GRU_2/while/gru_cell_27/MatMul:product:0(GRU_2/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_27/BiasAdd
GRU_2/while/gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_27/Const
'GRU_2/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_27/split/split_dim
GRU_2/while/gru_cell_27/splitSplit0GRU_2/while/gru_cell_27/split/split_dim:output:0(GRU_2/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_27/splitн
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_27/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_27/MatMul_1й
!GRU_2/while/gru_cell_27/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_27/MatMul_1:product:0(GRU_2/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_27/BiasAdd_1
GRU_2/while/gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_27/Const_1Ё
)GRU_2/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_27/split_1/split_dimЫ
GRU_2/while/gru_cell_27/split_1SplitV*GRU_2/while/gru_cell_27/BiasAdd_1:output:0(GRU_2/while/gru_cell_27/Const_1:output:02GRU_2/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_27/split_1Ч
GRU_2/while/gru_cell_27/addAddV2&GRU_2/while/gru_cell_27/split:output:0(GRU_2/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add 
GRU_2/while/gru_cell_27/SigmoidSigmoidGRU_2/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_27/SigmoidЫ
GRU_2/while/gru_cell_27/add_1AddV2&GRU_2/while/gru_cell_27/split:output:1(GRU_2/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_1І
!GRU_2/while/gru_cell_27/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_27/Sigmoid_1Ф
GRU_2/while/gru_cell_27/mulMul%GRU_2/while/gru_cell_27/Sigmoid_1:y:0(GRU_2/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mulТ
GRU_2/while/gru_cell_27/add_2AddV2&GRU_2/while/gru_cell_27/split:output:2GRU_2/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_2
GRU_2/while/gru_cell_27/TanhTanh!GRU_2/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/TanhЗ
GRU_2/while/gru_cell_27/mul_1Mul#GRU_2/while/gru_cell_27/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_1
GRU_2/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_27/sub/xР
GRU_2/while/gru_cell_27/subSub&GRU_2/while/gru_cell_27/sub/x:output:0#GRU_2/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/subК
GRU_2/while/gru_cell_27/mul_2MulGRU_2/while/gru_cell_27/sub:z:0 GRU_2/while/gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/mul_2П
GRU_2/while/gru_cell_27/add_3AddV2!GRU_2/while/gru_cell_27/mul_1:z:0!GRU_2/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_27/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_2/while/TensorArrayV2Write/TensorListSetItemh
GRU_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add/y
GRU_2/while/addAddV2gru_2_while_placeholderGRU_2/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/addl
GRU_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/add_1/y
GRU_2/while/add_1AddV2$gru_2_while_gru_2_while_loop_counterGRU_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_2/while/add_1p
GRU_2/while/IdentityIdentityGRU_2/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity
GRU_2/while/Identity_1Identity*gru_2_while_gru_2_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_2/while/Identity_1r
GRU_2/while/Identity_2IdentityGRU_2/while/add:z:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_2
GRU_2/while/Identity_3Identity@GRU_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_2/while/Identity_3
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_27/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_27_matmul_1_readvariableop_resource:gru_2_while_gru_cell_27_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_27_matmul_readvariableop_resource8gru_2_while_gru_cell_27_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_27_readvariableop_resource1gru_2_while_gru_cell_27_readvariableop_resource_0"5
gru_2_while_identityGRU_2/while/Identity:output:0"9
gru_2_while_identity_1GRU_2/while/Identity_1:output:0"9
gru_2_while_identity_2GRU_2/while/Identity_2:output:0"9
gru_2_while_identity_3GRU_2/while/Identity_3:output:0"9
gru_2_while_identity_4GRU_2/while/Identity_4:output:0"Р
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
а
Њ
while_cond_692847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_692847___redundant_placeholder04
0while_while_cond_692847___redundant_placeholder14
0while_while_cond_692847___redundant_placeholder24
0while_while_cond_692847___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
шW
ѓ
A__inference_GRU_2_layer_call_and_return_conditional_losses_696991

inputs'
#gru_cell_27_readvariableop_resource.
*gru_cell_27_matmul_readvariableop_resource0
,gru_cell_27_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_27/ReadVariableOp
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_27/unstackБ
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_27/MatMul/ReadVariableOpЉ
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMulЃ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAddh
gru_cell_27/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_27/Const
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split/split_dimм
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/splitЗ
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_27/MatMul_1/ReadVariableOpЅ
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/MatMul_1Љ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_27/BiasAdd_1
gru_cell_27/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_27/Const_1
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_27/split_1/split_dim
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const_1:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_27/split_1
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add|
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_1
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Sigmoid_1
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_2u
gru_cell_27/TanhTanhgru_cell_27/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/Tanh
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_1k
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_27/sub/x
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/sub
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/mul_2
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_27/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_696901*
condR
while_cond_696900*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

GRU_1_while_cond_695497(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_695497___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_695497___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_695497___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_695497___redundant_placeholder3
gru_1_while_identity

GRU_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
GRU_1/while/Lesso
GRU_1/while/IdentityIdentityGRU_1/while/Less:z:0*
T0
*
_output_shapes
: 2
GRU_1/while/Identity"5
gru_1_while_identityGRU_1/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
нH
з
GRU_1_while_body_695157(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_26_readvariableop_resource_0<
8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_26_readvariableop_resource:
6gru_1_while_gru_cell_26_matmul_readvariableop_resource<
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resourceЯ
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=GRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/GRU_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFGRU_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/GRU_1/while/TensorArrayV2Read/TensorListGetItemТ
&GRU_1/while/gru_cell_26/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_26/ReadVariableOpВ
GRU_1/while/gru_cell_26/unstackUnpack.GRU_1/while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_26/unstackз
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_26/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_26/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_26/MatMulг
GRU_1/while/gru_cell_26/BiasAddBiasAdd(GRU_1/while/gru_cell_26/MatMul:product:0(GRU_1/while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_26/BiasAdd
GRU_1/while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_26/Const
'GRU_1/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_26/split/split_dim
GRU_1/while/gru_cell_26/splitSplit0GRU_1/while/gru_cell_26/split/split_dim:output:0(GRU_1/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_26/splitн
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_26/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_26/MatMul_1й
!GRU_1/while/gru_cell_26/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_26/MatMul_1:product:0(GRU_1/while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_26/BiasAdd_1
GRU_1/while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_26/Const_1Ё
)GRU_1/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_26/split_1/split_dimЫ
GRU_1/while/gru_cell_26/split_1SplitV*GRU_1/while/gru_cell_26/BiasAdd_1:output:0(GRU_1/while/gru_cell_26/Const_1:output:02GRU_1/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_26/split_1Ч
GRU_1/while/gru_cell_26/addAddV2&GRU_1/while/gru_cell_26/split:output:0(GRU_1/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add 
GRU_1/while/gru_cell_26/SigmoidSigmoidGRU_1/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_26/SigmoidЫ
GRU_1/while/gru_cell_26/add_1AddV2&GRU_1/while/gru_cell_26/split:output:1(GRU_1/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_1І
!GRU_1/while/gru_cell_26/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_26/Sigmoid_1Ф
GRU_1/while/gru_cell_26/mulMul%GRU_1/while/gru_cell_26/Sigmoid_1:y:0(GRU_1/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mulТ
GRU_1/while/gru_cell_26/add_2AddV2&GRU_1/while/gru_cell_26/split:output:2GRU_1/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_2
GRU_1/while/gru_cell_26/TanhTanh!GRU_1/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/TanhЗ
GRU_1/while/gru_cell_26/mul_1Mul#GRU_1/while/gru_cell_26/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_1
GRU_1/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_26/sub/xР
GRU_1/while/gru_cell_26/subSub&GRU_1/while/gru_cell_26/sub/x:output:0#GRU_1/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/subК
GRU_1/while/gru_cell_26/mul_2MulGRU_1/while/gru_cell_26/sub:z:0 GRU_1/while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/mul_2П
GRU_1/while/gru_cell_26/add_3AddV2!GRU_1/while/gru_cell_26/mul_1:z:0!GRU_1/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_26/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype022
0GRU_1/while/TensorArrayV2Write/TensorListSetItemh
GRU_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add/y
GRU_1/while/addAddV2gru_1_while_placeholderGRU_1/while/add/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/addl
GRU_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/add_1/y
GRU_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_counterGRU_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
GRU_1/while/add_1p
GRU_1/while/IdentityIdentityGRU_1/while/add_1:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity
GRU_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations*
T0*
_output_shapes
: 2
GRU_1/while/Identity_1r
GRU_1/while/Identity_2IdentityGRU_1/while/add:z:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_2
GRU_1/while/Identity_3Identity@GRU_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
GRU_1/while/Identity_3
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_26_matmul_1_readvariableop_resource:gru_1_while_gru_cell_26_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_26_matmul_readvariableop_resource8gru_1_while_gru_cell_26_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_26_readvariableop_resource1gru_1_while_gru_cell_26_readvariableop_resource_0"5
gru_1_while_identityGRU_1/while/Identity:output:0"9
gru_1_while_identity_1GRU_1/while/Identity_1:output:0"9
gru_1_while_identity_2GRU_1/while/Identity_2:output:0"9
gru_1_while_identity_3GRU_1/while/Identity_3:output:0"9
gru_1_while_identity_4GRU_1/while/Identity_4:output:0"Р
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
шW
ѓ
A__inference_GRU_1_layer_call_and_return_conditional_losses_696470

inputs'
#gru_cell_26_readvariableop_resource.
*gru_cell_26_matmul_readvariableop_resource0
,gru_cell_26_matmul_1_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_26/ReadVariableOp
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_26/unstackБ
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_26/MatMul/ReadVariableOpЉ
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMulЃ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAddh
gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_26/Const
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split/split_dimм
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/splitЗ
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_26/MatMul_1/ReadVariableOpЅ
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/MatMul_1Љ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_26/BiasAdd_1
gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_26/Const_1
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_26/split_1/split_dim
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const_1:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_26/split_1
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add|
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_1
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Sigmoid_1
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_2u
gru_cell_26/TanhTanhgru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/Tanh
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_1k
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_26/sub/x
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/sub
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/mul_2
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_26/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_696380*
condR
while_cond_696379*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
Њ
while_cond_696719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_696719___redundant_placeholder04
0while_while_cond_696719___redundant_placeholder14
0while_while_cond_696719___redundant_placeholder24
0while_while_cond_696719___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
@
Е
while_body_695881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_26_readvariableop_resource_06
2while_gru_cell_26_matmul_readvariableop_resource_08
4while_gru_cell_26_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_26_readvariableop_resource4
0while_gru_cell_26_matmul_readvariableop_resource6
2while_gru_cell_26_matmul_1_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemА
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_26/ReadVariableOp 
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_26/unstackХ
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_26/MatMul/ReadVariableOpг
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMulЛ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAddt
while/gru_cell_26/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_26/Const
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_26/split/split_dimє
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/splitЫ
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_26/MatMul_1/ReadVariableOpМ
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/MatMul_1С
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_26/BiasAdd_1
while/gru_cell_26/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_26/Const_1
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_26/split_1/split_dim­
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0"while/gru_cell_26/Const_1:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_26/split_1Џ
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/SigmoidГ
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_1
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Sigmoid_1Ќ
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mulЊ
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_2
while/gru_cell_26/TanhTanhwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/Tanh
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_1w
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_26/sub/xЈ
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/subЂ
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0while/gru_cell_26/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/mul_2Ї
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_26/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: "ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ж
serving_defaultЂ
G
GRU_1_input8
serving_default_GRU_1_input:0џџџџџџџџџ;
OUT4
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ћз
+
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
N_default_save_signature
O__call__
*P&call_and_return_all_conditional_losses"ъ(
_tf_keras_sequentialЫ({"class_name": "Sequential", "name": "Supervisor", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Supervisor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "GRU_1_input"}}, {"class_name": "GRU", "config": {"name": "GRU_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "GRU", "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "OUT", "trainable": true, "dtype": "float32", "units": 20, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Supervisor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "GRU_1_input"}}, {"class_name": "GRU", "config": {"name": "GRU_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "GRU", "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "OUT", "trainable": true, "dtype": "float32", "units": 20, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
м
	cell

_inbound_nodes

state_spec
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"

_tf_keras_rnn_layerь	{"class_name": "GRU", "name": "GRU_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GRU_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}
м
cell
_inbound_nodes

state_spec
_outbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"

_tf_keras_rnn_layerь	{"class_name": "GRU", "name": "GRU_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}

_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЏ{"class_name": "Dense", "name": "OUT", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "OUT", "trainable": true, "dtype": "float32", "units": 20, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}
X
 0
!1
"2
#3
$4
%5
6
7"
trackable_list_wrapper
X
 0
!1
"2
#3
$4
%5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
&non_trainable_variables
'layer_regularization_losses
	variables
trainable_variables
(layer_metrics
regularization_losses

)layers
*metrics
O__call__
N_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
Ђ

 kernel
!recurrent_kernel
"bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_26", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й
/non_trainable_variables
0layer_regularization_losses
	variables
trainable_variables
1layer_metrics
regularization_losses

2states

3layers
4metrics
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
Ђ

#kernel
$recurrent_kernel
%bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_27", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й
9non_trainable_variables
:layer_regularization_losses
	variables
trainable_variables
;layer_metrics
regularization_losses

<states

=layers
>metrics
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2
OUT/kernel
:2OUT/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables
@layer_regularization_losses
	variables
trainable_variables
Alayer_metrics
regularization_losses

Blayers
Cmetrics
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
*:(<2GRU_1/gru_cell_26/kernel
4:2<2"GRU_1/gru_cell_26/recurrent_kernel
(:&<2GRU_1/gru_cell_26/bias
*:(<2GRU_2/gru_cell_27/kernel
4:2<2"GRU_2/gru_cell_27/recurrent_kernel
(:&<2GRU_2/gru_cell_27/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables
Elayer_regularization_losses
+	variables
,trainable_variables
Flayer_metrics
-regularization_losses

Glayers
Hmetrics
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables
Jlayer_regularization_losses
5	variables
6trainable_variables
Klayer_metrics
7regularization_losses

Llayers
Mmetrics
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ч2ф
!__inference__wrapped_model_692359О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
GRU_1_inputџџџџџџџџџ
њ2ї
+__inference_Supervisor_layer_call_fn_695812
+__inference_Supervisor_layer_call_fn_695088
+__inference_Supervisor_layer_call_fn_695791
+__inference_Supervisor_layer_call_fn_695067Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
F__inference_Supervisor_layer_call_and_return_conditional_losses_695046
F__inference_Supervisor_layer_call_and_return_conditional_losses_695770
F__inference_Supervisor_layer_call_and_return_conditional_losses_694705
F__inference_Supervisor_layer_call_and_return_conditional_losses_695429Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ћ2ј
&__inference_GRU_1_layer_call_fn_696141
&__inference_GRU_1_layer_call_fn_696492
&__inference_GRU_1_layer_call_fn_696481
&__inference_GRU_1_layer_call_fn_696152е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ч2ф
A__inference_GRU_1_layer_call_and_return_conditional_losses_695971
A__inference_GRU_1_layer_call_and_return_conditional_losses_696130
A__inference_GRU_1_layer_call_and_return_conditional_losses_696311
A__inference_GRU_1_layer_call_and_return_conditional_losses_696470е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ћ2ј
&__inference_GRU_2_layer_call_fn_696821
&__inference_GRU_2_layer_call_fn_696832
&__inference_GRU_2_layer_call_fn_697172
&__inference_GRU_2_layer_call_fn_697161е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ч2ф
A__inference_GRU_2_layer_call_and_return_conditional_losses_696810
A__inference_GRU_2_layer_call_and_return_conditional_losses_696651
A__inference_GRU_2_layer_call_and_return_conditional_losses_697150
A__inference_GRU_2_layer_call_and_return_conditional_losses_696991е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
$__inference_OUT_layer_call_fn_697212Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_OUT_layer_call_and_return_conditional_losses_697203Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
7B5
$__inference_signature_wrapper_694364GRU_1_input
 2
,__inference_gru_cell_26_layer_call_fn_697306
,__inference_gru_cell_26_layer_call_fn_697320О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_697252
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_697292О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
,__inference_gru_cell_27_layer_call_fn_697414
,__inference_gru_cell_27_layer_call_fn_697428О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_697360
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_697400О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 а
A__inference_GRU_1_layer_call_and_return_conditional_losses_695971" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 а
A__inference_GRU_1_layer_call_and_return_conditional_losses_696130" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Ж
A__inference_GRU_1_layer_call_and_return_conditional_losses_696311q" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Ж
A__inference_GRU_1_layer_call_and_return_conditional_losses_696470q" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Ї
&__inference_GRU_1_layer_call_fn_696141}" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЇ
&__inference_GRU_1_layer_call_fn_696152}" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџ
&__inference_GRU_1_layer_call_fn_696481d" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
&__inference_GRU_1_layer_call_fn_696492d" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџа
A__inference_GRU_2_layer_call_and_return_conditional_losses_696651%#$OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 а
A__inference_GRU_2_layer_call_and_return_conditional_losses_696810%#$OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Ж
A__inference_GRU_2_layer_call_and_return_conditional_losses_696991q%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Ж
A__inference_GRU_2_layer_call_and_return_conditional_losses_697150q%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Ї
&__inference_GRU_2_layer_call_fn_696821}%#$OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЇ
&__inference_GRU_2_layer_call_fn_696832}%#$OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџ
&__inference_GRU_2_layer_call_fn_697161d%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
&__inference_GRU_2_layer_call_fn_697172d%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЇ
?__inference_OUT_layer_call_and_return_conditional_losses_697203d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
$__inference_OUT_layer_call_fn_697212W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџС
F__inference_Supervisor_layer_call_and_return_conditional_losses_694705w" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 С
F__inference_Supervisor_layer_call_and_return_conditional_losses_695046w" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 М
F__inference_Supervisor_layer_call_and_return_conditional_losses_695429r" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 М
F__inference_Supervisor_layer_call_and_return_conditional_losses_695770r" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 
+__inference_Supervisor_layer_call_fn_695067j" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
+__inference_Supervisor_layer_call_fn_695088j" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
+__inference_Supervisor_layer_call_fn_695791e" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
+__inference_Supervisor_layer_call_fn_695812e" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
!__inference__wrapped_model_692359s" !%#$8Ђ5
.Ђ+
)&
GRU_1_inputџџџџџџџџџ
Њ "-Њ*
(
OUT!
OUTџџџџџџџџџ
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_697252З" !\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_697292З" !\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 к
,__inference_gru_cell_26_layer_call_fn_697306Љ" !\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџк
,__inference_gru_cell_26_layer_call_fn_697320Љ" !\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџ
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_697360З%#$\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_697400З%#$\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 к
,__inference_gru_cell_27_layer_call_fn_697414Љ%#$\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџк
,__inference_gru_cell_27_layer_call_fn_697428Љ%#$\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџЋ
$__inference_signature_wrapper_694364" !%#$GЂD
Ђ 
=Њ:
8
GRU_1_input)&
GRU_1_inputџџџџџџџџџ"-Њ*
(
OUT!
OUTџџџџџџџџџ