щП(
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
 "serve*2.3.12v2.3.0-54-gfcc4b966f18хп&
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
GRU_1/gru_cell_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameGRU_1/gru_cell_82/kernel

,GRU_1/gru_cell_82/kernel/Read/ReadVariableOpReadVariableOpGRU_1/gru_cell_82/kernel*
_output_shapes

:<*
dtype0
 
"GRU_1/gru_cell_82/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*3
shared_name$"GRU_1/gru_cell_82/recurrent_kernel

6GRU_1/gru_cell_82/recurrent_kernel/Read/ReadVariableOpReadVariableOp"GRU_1/gru_cell_82/recurrent_kernel*
_output_shapes

:<*
dtype0

GRU_1/gru_cell_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameGRU_1/gru_cell_82/bias

*GRU_1/gru_cell_82/bias/Read/ReadVariableOpReadVariableOpGRU_1/gru_cell_82/bias*
_output_shapes

:<*
dtype0

GRU_2/gru_cell_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameGRU_2/gru_cell_83/kernel

,GRU_2/gru_cell_83/kernel/Read/ReadVariableOpReadVariableOpGRU_2/gru_cell_83/kernel*
_output_shapes

:<*
dtype0
 
"GRU_2/gru_cell_83/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*3
shared_name$"GRU_2/gru_cell_83/recurrent_kernel

6GRU_2/gru_cell_83/recurrent_kernel/Read/ReadVariableOpReadVariableOp"GRU_2/gru_cell_83/recurrent_kernel*
_output_shapes

:<*
dtype0

GRU_2/gru_cell_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameGRU_2/gru_cell_83/bias

*GRU_2/gru_cell_83/bias/Read/ReadVariableOpReadVariableOpGRU_2/gru_cell_83/bias*
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
regularization_losses
	variables
trainable_variables
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
regularization_losses
	variables
trainable_variables
	keras_api

cell
_inbound_nodes

state_spec
_outbound_nodes
regularization_losses
	variables
trainable_variables
	keras_api
|
_inbound_nodes

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
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
­
regularization_losses

&layers
	variables
'layer_metrics
(non_trainable_variables
)metrics
*layer_regularization_losses
trainable_variables
 
~

 kernel
!recurrent_kernel
"bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
 
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
Й
regularization_losses

/layers

0states
	variables
1layer_metrics
2non_trainable_variables
3metrics
4layer_regularization_losses
trainable_variables
~

#kernel
$recurrent_kernel
%bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
 
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
Й
regularization_losses

9layers

:states
	variables
;layer_metrics
<non_trainable_variables
=metrics
>layer_regularization_losses
trainable_variables
 
VT
VARIABLE_VALUE
OUT/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEOUT/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

?layers
	variables
@layer_metrics
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
trainable_variables
TR
VARIABLE_VALUEGRU_1/gru_cell_82/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"GRU_1/gru_cell_82/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEGRU_1/gru_cell_82/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEGRU_2/gru_cell_83/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"GRU_2/gru_cell_83/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEGRU_2/gru_cell_83/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
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
­
+regularization_losses

Dlayers
,	variables
Elayer_metrics
Fnon_trainable_variables
Gmetrics
Hlayer_regularization_losses
-trainable_variables

	0
 
 
 
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
­
5regularization_losses

Ilayers
6	variables
Jlayer_metrics
Knon_trainable_variables
Lmetrics
Mlayer_regularization_losses
7trainable_variables
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

StatefulPartitionedCallStatefulPartitionedCallserving_default_GRU_1_inputGRU_1/gru_cell_82/biasGRU_1/gru_cell_82/kernel"GRU_1/gru_cell_82/recurrent_kernelGRU_2/gru_cell_83/biasGRU_2/gru_cell_83/kernel"GRU_2/gru_cell_83/recurrent_kernel
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1686688
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameOUT/kernel/Read/ReadVariableOpOUT/bias/Read/ReadVariableOp,GRU_1/gru_cell_82/kernel/Read/ReadVariableOp6GRU_1/gru_cell_82/recurrent_kernel/Read/ReadVariableOp*GRU_1/gru_cell_82/bias/Read/ReadVariableOp,GRU_2/gru_cell_83/kernel/Read/ReadVariableOp6GRU_2/gru_cell_83/recurrent_kernel/Read/ReadVariableOp*GRU_2/gru_cell_83/bias/Read/ReadVariableOpConst*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1689799
с
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
OUT/kernelOUT/biasGRU_1/gru_cell_82/kernel"GRU_1/gru_cell_82/recurrent_kernelGRU_1/gru_cell_82/biasGRU_2/gru_cell_83/kernel"GRU_2/gru_cell_83/recurrent_kernelGRU_2/gru_cell_83/bias*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1689833кІ&
@
Ж
while_body_1689044
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_83_readvariableop_resource_06
2while_gru_cell_83_matmul_readvariableop_resource_08
4while_gru_cell_83_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_83_readvariableop_resource4
0while_gru_cell_83_matmul_readvariableop_resource6
2while_gru_cell_83_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_83/ReadVariableOpReadVariableOp+while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_83/ReadVariableOp 
while/gru_cell_83/unstackUnpack(while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_83/unstackХ
'while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_83/MatMul/ReadVariableOpг
while/gru_cell_83/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMulЛ
while/gru_cell_83/BiasAddBiasAdd"while/gru_cell_83/MatMul:product:0"while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAddt
while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_83/Const
!while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_83/split/split_dimє
while/gru_cell_83/splitSplit*while/gru_cell_83/split/split_dim:output:0"while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/splitЫ
)while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_83/MatMul_1/ReadVariableOpМ
while/gru_cell_83/MatMul_1MatMulwhile_placeholder_21while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMul_1С
while/gru_cell_83/BiasAdd_1BiasAdd$while/gru_cell_83/MatMul_1:product:0"while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAdd_1
while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_83/Const_1
#while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_83/split_1/split_dim­
while/gru_cell_83/split_1SplitV$while/gru_cell_83/BiasAdd_1:output:0"while/gru_cell_83/Const_1:output:0,while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/split_1Џ
while/gru_cell_83/addAddV2 while/gru_cell_83/split:output:0"while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add
while/gru_cell_83/SigmoidSigmoidwhile/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/SigmoidГ
while/gru_cell_83/add_1AddV2 while/gru_cell_83/split:output:1"while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_1
while/gru_cell_83/Sigmoid_1Sigmoidwhile/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Sigmoid_1Ќ
while/gru_cell_83/mulMulwhile/gru_cell_83/Sigmoid_1:y:0"while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mulЊ
while/gru_cell_83/add_2AddV2 while/gru_cell_83/split:output:2while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_2
while/gru_cell_83/TanhTanhwhile/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Tanh
while/gru_cell_83/mul_1Mulwhile/gru_cell_83/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_1w
while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_83/sub/xЈ
while/gru_cell_83/subSub while/gru_cell_83/sub/x:output:0while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/subЂ
while/gru_cell_83/mul_2Mulwhile/gru_cell_83/sub:z:0while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_2Ї
while/gru_cell_83/add_3AddV2while/gru_cell_83/mul_1:z:0while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_83/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_83_matmul_1_readvariableop_resource4while_gru_cell_83_matmul_1_readvariableop_resource_0"f
0while_gru_cell_83_matmul_readvariableop_resource2while_gru_cell_83_matmul_readvariableop_resource_0"X
)while_gru_cell_83_readvariableop_resource+while_gru_cell_83_readvariableop_resource_0")
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
Ф
ђ
#Supervisor_GRU_2_while_cond_1684565>
:supervisor_gru_2_while_supervisor_gru_2_while_loop_counterD
@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations&
"supervisor_gru_2_while_placeholder(
$supervisor_gru_2_while_placeholder_1(
$supervisor_gru_2_while_placeholder_2@
<supervisor_gru_2_while_less_supervisor_gru_2_strided_slice_1W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1684565___redundant_placeholder0W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1684565___redundant_placeholder1W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1684565___redundant_placeholder2W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1684565___redundant_placeholder3#
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
	
Ё
GRU_1_while_cond_1686756(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1686756___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1686756___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1686756___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1686756___redundant_placeholder3
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
оH
и
GRU_1_while_body_1686757(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_82_readvariableop_resource_0<
8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_82_readvariableop_resource:
6gru_1_while_gru_cell_82_matmul_readvariableop_resource<
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_82/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_82/ReadVariableOpВ
GRU_1/while/gru_cell_82/unstackUnpack.GRU_1/while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_82/unstackз
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_82/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_82/MatMulг
GRU_1/while/gru_cell_82/BiasAddBiasAdd(GRU_1/while/gru_cell_82/MatMul:product:0(GRU_1/while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_82/BiasAdd
GRU_1/while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_82/Const
'GRU_1/while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_82/split/split_dim
GRU_1/while/gru_cell_82/splitSplit0GRU_1/while/gru_cell_82/split/split_dim:output:0(GRU_1/while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_82/splitн
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_82/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_82/MatMul_1й
!GRU_1/while/gru_cell_82/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_82/MatMul_1:product:0(GRU_1/while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_82/BiasAdd_1
GRU_1/while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_82/Const_1Ё
)GRU_1/while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_82/split_1/split_dimЫ
GRU_1/while/gru_cell_82/split_1SplitV*GRU_1/while/gru_cell_82/BiasAdd_1:output:0(GRU_1/while/gru_cell_82/Const_1:output:02GRU_1/while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_82/split_1Ч
GRU_1/while/gru_cell_82/addAddV2&GRU_1/while/gru_cell_82/split:output:0(GRU_1/while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add 
GRU_1/while/gru_cell_82/SigmoidSigmoidGRU_1/while/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_82/SigmoidЫ
GRU_1/while/gru_cell_82/add_1AddV2&GRU_1/while/gru_cell_82/split:output:1(GRU_1/while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_1І
!GRU_1/while/gru_cell_82/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_82/Sigmoid_1Ф
GRU_1/while/gru_cell_82/mulMul%GRU_1/while/gru_cell_82/Sigmoid_1:y:0(GRU_1/while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mulТ
GRU_1/while/gru_cell_82/add_2AddV2&GRU_1/while/gru_cell_82/split:output:2GRU_1/while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_2
GRU_1/while/gru_cell_82/TanhTanh!GRU_1/while/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/TanhЗ
GRU_1/while/gru_cell_82/mul_1Mul#GRU_1/while/gru_cell_82/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_1
GRU_1/while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_82/sub/xР
GRU_1/while/gru_cell_82/subSub&GRU_1/while/gru_cell_82/sub/x:output:0#GRU_1/while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/subК
GRU_1/while/gru_cell_82/mul_2MulGRU_1/while/gru_cell_82/sub:z:0 GRU_1/while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_2П
GRU_1/while/gru_cell_82/add_3AddV2!GRU_1/while/gru_cell_82/mul_1:z:0!GRU_1/while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_82/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resource:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_82_matmul_readvariableop_resource8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_82_readvariableop_resource1gru_1_while_gru_cell_82_readvariableop_resource_0"5
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
ыW
є
B__inference_GRU_2_layer_call_and_return_conditional_losses_1686475

inputs'
#gru_cell_83_readvariableop_resource.
*gru_cell_83_matmul_readvariableop_resource0
,gru_cell_83_matmul_1_readvariableop_resource
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
gru_cell_83/ReadVariableOpReadVariableOp#gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_83/ReadVariableOp
gru_cell_83/unstackUnpack"gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_83/unstackБ
!gru_cell_83/MatMul/ReadVariableOpReadVariableOp*gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_83/MatMul/ReadVariableOpЉ
gru_cell_83/MatMulMatMulstrided_slice_2:output:0)gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMulЃ
gru_cell_83/BiasAddBiasAddgru_cell_83/MatMul:product:0gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAddh
gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_83/Const
gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split/split_dimм
gru_cell_83/splitSplit$gru_cell_83/split/split_dim:output:0gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/splitЗ
#gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_83/MatMul_1/ReadVariableOpЅ
gru_cell_83/MatMul_1MatMulzeros:output:0+gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMul_1Љ
gru_cell_83/BiasAdd_1BiasAddgru_cell_83/MatMul_1:product:0gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAdd_1
gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_83/Const_1
gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split_1/split_dim
gru_cell_83/split_1SplitVgru_cell_83/BiasAdd_1:output:0gru_cell_83/Const_1:output:0&gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/split_1
gru_cell_83/addAddV2gru_cell_83/split:output:0gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add|
gru_cell_83/SigmoidSigmoidgru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid
gru_cell_83/add_1AddV2gru_cell_83/split:output:1gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_1
gru_cell_83/Sigmoid_1Sigmoidgru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid_1
gru_cell_83/mulMulgru_cell_83/Sigmoid_1:y:0gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul
gru_cell_83/add_2AddV2gru_cell_83/split:output:2gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_2u
gru_cell_83/TanhTanhgru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Tanh
gru_cell_83/mul_1Mulgru_cell_83/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_1k
gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_83/sub/x
gru_cell_83/subSubgru_cell_83/sub/x:output:0gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/sub
gru_cell_83/mul_2Mulgru_cell_83/sub:z:0gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_2
gru_cell_83/add_3AddV2gru_cell_83/mul_1:z:0gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_83_readvariableop_resource*gru_cell_83_matmul_readvariableop_resource,gru_cell_83_matmul_1_readvariableop_resource*
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
bodyR
while_body_1686385*
condR
while_cond_1686384*8
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
р	
Џ
-__inference_gru_cell_82_layer_call_fn_1689630

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЄ
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_16847552
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
ЊX
і
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688454
inputs_0'
#gru_cell_82_readvariableop_resource.
*gru_cell_82_matmul_readvariableop_resource0
,gru_cell_82_matmul_1_readvariableop_resource
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
gru_cell_82/ReadVariableOpReadVariableOp#gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_82/ReadVariableOp
gru_cell_82/unstackUnpack"gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_82/unstackБ
!gru_cell_82/MatMul/ReadVariableOpReadVariableOp*gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_82/MatMul/ReadVariableOpЉ
gru_cell_82/MatMulMatMulstrided_slice_2:output:0)gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMulЃ
gru_cell_82/BiasAddBiasAddgru_cell_82/MatMul:product:0gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAddh
gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_82/Const
gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split/split_dimм
gru_cell_82/splitSplit$gru_cell_82/split/split_dim:output:0gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/splitЗ
#gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_82/MatMul_1/ReadVariableOpЅ
gru_cell_82/MatMul_1MatMulzeros:output:0+gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMul_1Љ
gru_cell_82/BiasAdd_1BiasAddgru_cell_82/MatMul_1:product:0gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAdd_1
gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_82/Const_1
gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split_1/split_dim
gru_cell_82/split_1SplitVgru_cell_82/BiasAdd_1:output:0gru_cell_82/Const_1:output:0&gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/split_1
gru_cell_82/addAddV2gru_cell_82/split:output:0gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add|
gru_cell_82/SigmoidSigmoidgru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid
gru_cell_82/add_1AddV2gru_cell_82/split:output:1gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_1
gru_cell_82/Sigmoid_1Sigmoidgru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid_1
gru_cell_82/mulMulgru_cell_82/Sigmoid_1:y:0gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul
gru_cell_82/add_2AddV2gru_cell_82/split:output:2gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_2u
gru_cell_82/TanhTanhgru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Tanh
gru_cell_82/mul_1Mulgru_cell_82/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_1k
gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_82/sub/x
gru_cell_82/subSubgru_cell_82/sub/x:output:0gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/sub
gru_cell_82/mul_2Mulgru_cell_82/sub:z:0gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_2
gru_cell_82/add_3AddV2gru_cell_82/mul_1:z:0gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_82_readvariableop_resource*gru_cell_82_matmul_readvariableop_resource,gru_cell_82_matmul_1_readvariableop_resource*
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
bodyR
while_body_1688364*
condR
while_cond_1688363*8
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
џ!
т
while_body_1685616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_83_1685638_0
while_gru_cell_83_1685640_0
while_gru_cell_83_1685642_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_83_1685638
while_gru_cell_83_1685640
while_gru_cell_83_1685642Ђ)while/gru_cell_83/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemЕ
)while/gru_cell_83/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_83_1685638_0while_gru_cell_83_1685640_0while_gru_cell_83_1685642_0*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_16853172+
)while/gru_cell_83/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_83/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_83/StatefulPartitionedCall:output:1*^while/gru_cell_83/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_83_1685638while_gru_cell_83_1685638_0"8
while_gru_cell_83_1685640while_gru_cell_83_1685640_0"8
while_gru_cell_83_1685642while_gru_cell_83_1685642_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_83/StatefulPartitionedCall)while/gru_cell_83/StatefulPartitionedCall: 
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
@
Ж
while_body_1689225
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_83_readvariableop_resource_06
2while_gru_cell_83_matmul_readvariableop_resource_08
4while_gru_cell_83_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_83_readvariableop_resource4
0while_gru_cell_83_matmul_readvariableop_resource6
2while_gru_cell_83_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_83/ReadVariableOpReadVariableOp+while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_83/ReadVariableOp 
while/gru_cell_83/unstackUnpack(while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_83/unstackХ
'while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_83/MatMul/ReadVariableOpг
while/gru_cell_83/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMulЛ
while/gru_cell_83/BiasAddBiasAdd"while/gru_cell_83/MatMul:product:0"while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAddt
while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_83/Const
!while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_83/split/split_dimє
while/gru_cell_83/splitSplit*while/gru_cell_83/split/split_dim:output:0"while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/splitЫ
)while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_83/MatMul_1/ReadVariableOpМ
while/gru_cell_83/MatMul_1MatMulwhile_placeholder_21while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMul_1С
while/gru_cell_83/BiasAdd_1BiasAdd$while/gru_cell_83/MatMul_1:product:0"while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAdd_1
while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_83/Const_1
#while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_83/split_1/split_dim­
while/gru_cell_83/split_1SplitV$while/gru_cell_83/BiasAdd_1:output:0"while/gru_cell_83/Const_1:output:0,while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/split_1Џ
while/gru_cell_83/addAddV2 while/gru_cell_83/split:output:0"while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add
while/gru_cell_83/SigmoidSigmoidwhile/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/SigmoidГ
while/gru_cell_83/add_1AddV2 while/gru_cell_83/split:output:1"while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_1
while/gru_cell_83/Sigmoid_1Sigmoidwhile/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Sigmoid_1Ќ
while/gru_cell_83/mulMulwhile/gru_cell_83/Sigmoid_1:y:0"while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mulЊ
while/gru_cell_83/add_2AddV2 while/gru_cell_83/split:output:2while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_2
while/gru_cell_83/TanhTanhwhile/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Tanh
while/gru_cell_83/mul_1Mulwhile/gru_cell_83/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_1w
while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_83/sub/xЈ
while/gru_cell_83/subSub while/gru_cell_83/sub/x:output:0while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/subЂ
while/gru_cell_83/mul_2Mulwhile/gru_cell_83/sub:z:0while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_2Ї
while/gru_cell_83/add_3AddV2while/gru_cell_83/mul_1:z:0while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_83/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_83_matmul_1_readvariableop_resource4while_gru_cell_83_matmul_1_readvariableop_resource_0"f
0while_gru_cell_83_matmul_readvariableop_resource2while_gru_cell_83_matmul_readvariableop_resource_0"X
)while_gru_cell_83_readvariableop_resource+while_gru_cell_83_readvariableop_resource_0")
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
ЊX
і
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689315
inputs_0'
#gru_cell_83_readvariableop_resource.
*gru_cell_83_matmul_readvariableop_resource0
,gru_cell_83_matmul_1_readvariableop_resource
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
gru_cell_83/ReadVariableOpReadVariableOp#gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_83/ReadVariableOp
gru_cell_83/unstackUnpack"gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_83/unstackБ
!gru_cell_83/MatMul/ReadVariableOpReadVariableOp*gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_83/MatMul/ReadVariableOpЉ
gru_cell_83/MatMulMatMulstrided_slice_2:output:0)gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMulЃ
gru_cell_83/BiasAddBiasAddgru_cell_83/MatMul:product:0gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAddh
gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_83/Const
gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split/split_dimм
gru_cell_83/splitSplit$gru_cell_83/split/split_dim:output:0gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/splitЗ
#gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_83/MatMul_1/ReadVariableOpЅ
gru_cell_83/MatMul_1MatMulzeros:output:0+gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMul_1Љ
gru_cell_83/BiasAdd_1BiasAddgru_cell_83/MatMul_1:product:0gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAdd_1
gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_83/Const_1
gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split_1/split_dim
gru_cell_83/split_1SplitVgru_cell_83/BiasAdd_1:output:0gru_cell_83/Const_1:output:0&gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/split_1
gru_cell_83/addAddV2gru_cell_83/split:output:0gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add|
gru_cell_83/SigmoidSigmoidgru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid
gru_cell_83/add_1AddV2gru_cell_83/split:output:1gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_1
gru_cell_83/Sigmoid_1Sigmoidgru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid_1
gru_cell_83/mulMulgru_cell_83/Sigmoid_1:y:0gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul
gru_cell_83/add_2AddV2gru_cell_83/split:output:2gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_2u
gru_cell_83/TanhTanhgru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Tanh
gru_cell_83/mul_1Mulgru_cell_83/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_1k
gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_83/sub/x
gru_cell_83/subSubgru_cell_83/sub/x:output:0gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/sub
gru_cell_83/mul_2Mulgru_cell_83/sub:z:0gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_2
gru_cell_83/add_3AddV2gru_cell_83/mul_1:z:0gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_83_readvariableop_resource*gru_cell_83_matmul_readvariableop_resource,gru_cell_83_matmul_1_readvariableop_resource*
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
bodyR
while_body_1689225*
condR
while_cond_1689224*8
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
	
Ё
GRU_2_while_cond_1686911(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1686911___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1686911___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1686911___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1686911___redundant_placeholder3
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
Т
р
,__inference_Supervisor_layer_call_fn_1687412
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЮ
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
GPU 2J 8 *P
fKRI
G__inference_Supervisor_layer_call_and_return_conditional_losses_16866462
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
е
Џ
while_cond_1689224
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1689224___redundant_placeholder05
1while_while_cond_1689224___redundant_placeholder15
1while_while_cond_1689224___redundant_placeholder25
1while_while_cond_1689224___redundant_placeholder3
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
е
Џ
while_cond_1686384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1686384___redundant_placeholder05
1while_while_cond_1686384___redundant_placeholder15
1while_while_cond_1686384___redundant_placeholder25
1while_while_cond_1686384___redundant_placeholder3
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
ыW
є
B__inference_GRU_2_layer_call_and_return_conditional_losses_1686316

inputs'
#gru_cell_83_readvariableop_resource.
*gru_cell_83_matmul_readvariableop_resource0
,gru_cell_83_matmul_1_readvariableop_resource
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
gru_cell_83/ReadVariableOpReadVariableOp#gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_83/ReadVariableOp
gru_cell_83/unstackUnpack"gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_83/unstackБ
!gru_cell_83/MatMul/ReadVariableOpReadVariableOp*gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_83/MatMul/ReadVariableOpЉ
gru_cell_83/MatMulMatMulstrided_slice_2:output:0)gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMulЃ
gru_cell_83/BiasAddBiasAddgru_cell_83/MatMul:product:0gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAddh
gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_83/Const
gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split/split_dimм
gru_cell_83/splitSplit$gru_cell_83/split/split_dim:output:0gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/splitЗ
#gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_83/MatMul_1/ReadVariableOpЅ
gru_cell_83/MatMul_1MatMulzeros:output:0+gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMul_1Љ
gru_cell_83/BiasAdd_1BiasAddgru_cell_83/MatMul_1:product:0gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAdd_1
gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_83/Const_1
gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split_1/split_dim
gru_cell_83/split_1SplitVgru_cell_83/BiasAdd_1:output:0gru_cell_83/Const_1:output:0&gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/split_1
gru_cell_83/addAddV2gru_cell_83/split:output:0gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add|
gru_cell_83/SigmoidSigmoidgru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid
gru_cell_83/add_1AddV2gru_cell_83/split:output:1gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_1
gru_cell_83/Sigmoid_1Sigmoidgru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid_1
gru_cell_83/mulMulgru_cell_83/Sigmoid_1:y:0gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul
gru_cell_83/add_2AddV2gru_cell_83/split:output:2gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_2u
gru_cell_83/TanhTanhgru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Tanh
gru_cell_83/mul_1Mulgru_cell_83/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_1k
gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_83/sub/x
gru_cell_83/subSubgru_cell_83/sub/x:output:0gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/sub
gru_cell_83/mul_2Mulgru_cell_83/sub:z:0gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_2
gru_cell_83/add_3AddV2gru_cell_83/mul_1:z:0gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_83_readvariableop_resource*gru_cell_83_matmul_readvariableop_resource,gru_cell_83_matmul_1_readvariableop_resource*
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
bodyR
while_body_1686226*
condR
while_cond_1686225*8
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
ЊX
і
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688295
inputs_0'
#gru_cell_82_readvariableop_resource.
*gru_cell_82_matmul_readvariableop_resource0
,gru_cell_82_matmul_1_readvariableop_resource
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
gru_cell_82/ReadVariableOpReadVariableOp#gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_82/ReadVariableOp
gru_cell_82/unstackUnpack"gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_82/unstackБ
!gru_cell_82/MatMul/ReadVariableOpReadVariableOp*gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_82/MatMul/ReadVariableOpЉ
gru_cell_82/MatMulMatMulstrided_slice_2:output:0)gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMulЃ
gru_cell_82/BiasAddBiasAddgru_cell_82/MatMul:product:0gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAddh
gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_82/Const
gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split/split_dimм
gru_cell_82/splitSplit$gru_cell_82/split/split_dim:output:0gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/splitЗ
#gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_82/MatMul_1/ReadVariableOpЅ
gru_cell_82/MatMul_1MatMulzeros:output:0+gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMul_1Љ
gru_cell_82/BiasAdd_1BiasAddgru_cell_82/MatMul_1:product:0gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAdd_1
gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_82/Const_1
gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split_1/split_dim
gru_cell_82/split_1SplitVgru_cell_82/BiasAdd_1:output:0gru_cell_82/Const_1:output:0&gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/split_1
gru_cell_82/addAddV2gru_cell_82/split:output:0gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add|
gru_cell_82/SigmoidSigmoidgru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid
gru_cell_82/add_1AddV2gru_cell_82/split:output:1gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_1
gru_cell_82/Sigmoid_1Sigmoidgru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid_1
gru_cell_82/mulMulgru_cell_82/Sigmoid_1:y:0gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul
gru_cell_82/add_2AddV2gru_cell_82/split:output:2gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_2u
gru_cell_82/TanhTanhgru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Tanh
gru_cell_82/mul_1Mulgru_cell_82/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_1k
gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_82/sub/x
gru_cell_82/subSubgru_cell_82/sub/x:output:0gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/sub
gru_cell_82/mul_2Mulgru_cell_82/sub:z:0gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_2
gru_cell_82/add_3AddV2gru_cell_82/mul_1:z:0gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_82_readvariableop_resource*gru_cell_82_matmul_readvariableop_resource,gru_cell_82_matmul_1_readvariableop_resource*
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
bodyR
while_body_1688205*
condR
while_cond_1688204*8
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
ЊX


#Supervisor_GRU_1_while_body_1684411>
:supervisor_gru_1_while_supervisor_gru_1_while_loop_counterD
@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations&
"supervisor_gru_1_while_placeholder(
$supervisor_gru_1_while_placeholder_1(
$supervisor_gru_1_while_placeholder_2=
9supervisor_gru_1_while_supervisor_gru_1_strided_slice_1_0y
usupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor_0@
<supervisor_gru_1_while_gru_cell_82_readvariableop_resource_0G
Csupervisor_gru_1_while_gru_cell_82_matmul_readvariableop_resource_0I
Esupervisor_gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0#
supervisor_gru_1_while_identity%
!supervisor_gru_1_while_identity_1%
!supervisor_gru_1_while_identity_2%
!supervisor_gru_1_while_identity_3%
!supervisor_gru_1_while_identity_4;
7supervisor_gru_1_while_supervisor_gru_1_strided_slice_1w
ssupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor>
:supervisor_gru_1_while_gru_cell_82_readvariableop_resourceE
Asupervisor_gru_1_while_gru_cell_82_matmul_readvariableop_resourceG
Csupervisor_gru_1_while_gru_cell_82_matmul_1_readvariableop_resourceх
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
1Supervisor/GRU_1/while/gru_cell_82/ReadVariableOpReadVariableOp<supervisor_gru_1_while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype023
1Supervisor/GRU_1/while/gru_cell_82/ReadVariableOpг
*Supervisor/GRU_1/while/gru_cell_82/unstackUnpack9Supervisor/GRU_1/while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2,
*Supervisor/GRU_1/while/gru_cell_82/unstackј
8Supervisor/GRU_1/while/gru_cell_82/MatMul/ReadVariableOpReadVariableOpCsupervisor_gru_1_while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02:
8Supervisor/GRU_1/while/gru_cell_82/MatMul/ReadVariableOp
)Supervisor/GRU_1/while/gru_cell_82/MatMulMatMulASupervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:0@Supervisor/GRU_1/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2+
)Supervisor/GRU_1/while/gru_cell_82/MatMulџ
*Supervisor/GRU_1/while/gru_cell_82/BiasAddBiasAdd3Supervisor/GRU_1/while/gru_cell_82/MatMul:product:03Supervisor/GRU_1/while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2,
*Supervisor/GRU_1/while/gru_cell_82/BiasAdd
(Supervisor/GRU_1/while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(Supervisor/GRU_1/while/gru_cell_82/ConstГ
2Supervisor/GRU_1/while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2Supervisor/GRU_1/while/gru_cell_82/split/split_dimИ
(Supervisor/GRU_1/while/gru_cell_82/splitSplit;Supervisor/GRU_1/while/gru_cell_82/split/split_dim:output:03Supervisor/GRU_1/while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2*
(Supervisor/GRU_1/while/gru_cell_82/splitў
:Supervisor/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOpEsupervisor_gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02<
:Supervisor/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOp
+Supervisor/GRU_1/while/gru_cell_82/MatMul_1MatMul$supervisor_gru_1_while_placeholder_2BSupervisor/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2-
+Supervisor/GRU_1/while/gru_cell_82/MatMul_1
,Supervisor/GRU_1/while/gru_cell_82/BiasAdd_1BiasAdd5Supervisor/GRU_1/while/gru_cell_82/MatMul_1:product:03Supervisor/GRU_1/while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2.
,Supervisor/GRU_1/while/gru_cell_82/BiasAdd_1­
*Supervisor/GRU_1/while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2,
*Supervisor/GRU_1/while/gru_cell_82/Const_1З
4Supervisor/GRU_1/while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ26
4Supervisor/GRU_1/while/gru_cell_82/split_1/split_dim
*Supervisor/GRU_1/while/gru_cell_82/split_1SplitV5Supervisor/GRU_1/while/gru_cell_82/BiasAdd_1:output:03Supervisor/GRU_1/while/gru_cell_82/Const_1:output:0=Supervisor/GRU_1/while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*Supervisor/GRU_1/while/gru_cell_82/split_1ѓ
&Supervisor/GRU_1/while/gru_cell_82/addAddV21Supervisor/GRU_1/while/gru_cell_82/split:output:03Supervisor/GRU_1/while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_82/addС
*Supervisor/GRU_1/while/gru_cell_82/SigmoidSigmoid*Supervisor/GRU_1/while/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*Supervisor/GRU_1/while/gru_cell_82/Sigmoidї
(Supervisor/GRU_1/while/gru_cell_82/add_1AddV21Supervisor/GRU_1/while/gru_cell_82/split:output:13Supervisor/GRU_1/while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_82/add_1Ч
,Supervisor/GRU_1/while/gru_cell_82/Sigmoid_1Sigmoid,Supervisor/GRU_1/while/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,Supervisor/GRU_1/while/gru_cell_82/Sigmoid_1№
&Supervisor/GRU_1/while/gru_cell_82/mulMul0Supervisor/GRU_1/while/gru_cell_82/Sigmoid_1:y:03Supervisor/GRU_1/while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_82/mulю
(Supervisor/GRU_1/while/gru_cell_82/add_2AddV21Supervisor/GRU_1/while/gru_cell_82/split:output:2*Supervisor/GRU_1/while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_82/add_2К
'Supervisor/GRU_1/while/gru_cell_82/TanhTanh,Supervisor/GRU_1/while/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'Supervisor/GRU_1/while/gru_cell_82/Tanhу
(Supervisor/GRU_1/while/gru_cell_82/mul_1Mul.Supervisor/GRU_1/while/gru_cell_82/Sigmoid:y:0$supervisor_gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_82/mul_1
(Supervisor/GRU_1/while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(Supervisor/GRU_1/while/gru_cell_82/sub/xь
&Supervisor/GRU_1/while/gru_cell_82/subSub1Supervisor/GRU_1/while/gru_cell_82/sub/x:output:0.Supervisor/GRU_1/while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_82/subц
(Supervisor/GRU_1/while/gru_cell_82/mul_2Mul*Supervisor/GRU_1/while/gru_cell_82/sub:z:0+Supervisor/GRU_1/while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_82/mul_2ы
(Supervisor/GRU_1/while/gru_cell_82/add_3AddV2,Supervisor/GRU_1/while/gru_cell_82/mul_1:z:0,Supervisor/GRU_1/while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_82/add_3Д
;Supervisor/GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$supervisor_gru_1_while_placeholder_1"supervisor_gru_1_while_placeholder,Supervisor/GRU_1/while/gru_cell_82/add_3:z:0*
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
!Supervisor/GRU_1/while/Identity_4Identity,Supervisor/GRU_1/while/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_1/while/Identity_4"
Csupervisor_gru_1_while_gru_cell_82_matmul_1_readvariableop_resourceEsupervisor_gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0"
Asupervisor_gru_1_while_gru_cell_82_matmul_readvariableop_resourceCsupervisor_gru_1_while_gru_cell_82_matmul_readvariableop_resource_0"z
:supervisor_gru_1_while_gru_cell_82_readvariableop_resource<supervisor_gru_1_while_gru_cell_82_readvariableop_resource_0"K
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
Й
и
G__inference_Supervisor_layer_call_and_return_conditional_losses_1686646

inputs
gru_1_1686626
gru_1_1686628
gru_1_1686630
gru_2_1686633
gru_2_1686635
gru_2_1686637
out_1686640
out_1686642
identityЂGRU_1/StatefulPartitionedCallЂGRU_2/StatefulPartitionedCallЂOUT/StatefulPartitionedCall
GRU_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_1686626gru_1_1686628gru_1_1686630*
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
GPU 2J 8 *K
fFRD
B__inference_GRU_1_layer_call_and_return_conditional_losses_16861282
GRU_1/StatefulPartitionedCallН
GRU_2/StatefulPartitionedCallStatefulPartitionedCall&GRU_1/StatefulPartitionedCall:output:0gru_2_1686633gru_2_1686635gru_2_1686637*
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
GPU 2J 8 *K
fFRD
B__inference_GRU_2_layer_call_and_return_conditional_losses_16864752
GRU_2/StatefulPartitionedCallЂ
OUT/StatefulPartitionedCallStatefulPartitionedCall&GRU_2/StatefulPartitionedCall:output:0out_1686640out_1686642*
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
GPU 2J 8 *I
fDRB
@__inference_OUT_layer_call_and_return_conditional_losses_16865362
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
і<
к
B__inference_GRU_2_layer_call_and_return_conditional_losses_1685798

inputs
gru_cell_83_1685722
gru_cell_83_1685724
gru_cell_83_1685726
identityЂ#gru_cell_83/StatefulPartitionedCallЂwhileD
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
strided_slice_2є
#gru_cell_83/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_83_1685722gru_cell_83_1685724gru_cell_83_1685726*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_16853572%
#gru_cell_83/StatefulPartitionedCall
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
while/loop_counterь
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_83_1685722gru_cell_83_1685724gru_cell_83_1685726*
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
bodyR
while_body_1685734*
condR
while_cond_1685733*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_83/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_83/StatefulPartitionedCall#gru_cell_83/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
@
Ж
while_body_1686038
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_82_readvariableop_resource_06
2while_gru_cell_82_matmul_readvariableop_resource_08
4while_gru_cell_82_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_82_readvariableop_resource4
0while_gru_cell_82_matmul_readvariableop_resource6
2while_gru_cell_82_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_82/ReadVariableOpReadVariableOp+while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_82/ReadVariableOp 
while/gru_cell_82/unstackUnpack(while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_82/unstackХ
'while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_82/MatMul/ReadVariableOpг
while/gru_cell_82/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMulЛ
while/gru_cell_82/BiasAddBiasAdd"while/gru_cell_82/MatMul:product:0"while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAddt
while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_82/Const
!while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_82/split/split_dimє
while/gru_cell_82/splitSplit*while/gru_cell_82/split/split_dim:output:0"while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/splitЫ
)while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_82/MatMul_1/ReadVariableOpМ
while/gru_cell_82/MatMul_1MatMulwhile_placeholder_21while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMul_1С
while/gru_cell_82/BiasAdd_1BiasAdd$while/gru_cell_82/MatMul_1:product:0"while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAdd_1
while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_82/Const_1
#while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_82/split_1/split_dim­
while/gru_cell_82/split_1SplitV$while/gru_cell_82/BiasAdd_1:output:0"while/gru_cell_82/Const_1:output:0,while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/split_1Џ
while/gru_cell_82/addAddV2 while/gru_cell_82/split:output:0"while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add
while/gru_cell_82/SigmoidSigmoidwhile/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/SigmoidГ
while/gru_cell_82/add_1AddV2 while/gru_cell_82/split:output:1"while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_1
while/gru_cell_82/Sigmoid_1Sigmoidwhile/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Sigmoid_1Ќ
while/gru_cell_82/mulMulwhile/gru_cell_82/Sigmoid_1:y:0"while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mulЊ
while/gru_cell_82/add_2AddV2 while/gru_cell_82/split:output:2while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_2
while/gru_cell_82/TanhTanhwhile/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Tanh
while/gru_cell_82/mul_1Mulwhile/gru_cell_82/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_1w
while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_82/sub/xЈ
while/gru_cell_82/subSub while/gru_cell_82/sub/x:output:0while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/subЂ
while/gru_cell_82/mul_2Mulwhile/gru_cell_82/sub:z:0while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_2Ї
while/gru_cell_82/add_3AddV2while/gru_cell_82/mul_1:z:0while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_82/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_82_matmul_1_readvariableop_resource4while_gru_cell_82_matmul_1_readvariableop_resource_0"f
0while_gru_cell_82_matmul_readvariableop_resource2while_gru_cell_82_matmul_readvariableop_resource_0"X
)while_gru_cell_82_readvariableop_resource+while_gru_cell_82_readvariableop_resource_0")
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
я
ь
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1689684

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
@
Ж
while_body_1685879
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_82_readvariableop_resource_06
2while_gru_cell_82_matmul_readvariableop_resource_08
4while_gru_cell_82_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_82_readvariableop_resource4
0while_gru_cell_82_matmul_readvariableop_resource6
2while_gru_cell_82_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_82/ReadVariableOpReadVariableOp+while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_82/ReadVariableOp 
while/gru_cell_82/unstackUnpack(while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_82/unstackХ
'while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_82/MatMul/ReadVariableOpг
while/gru_cell_82/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMulЛ
while/gru_cell_82/BiasAddBiasAdd"while/gru_cell_82/MatMul:product:0"while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAddt
while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_82/Const
!while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_82/split/split_dimє
while/gru_cell_82/splitSplit*while/gru_cell_82/split/split_dim:output:0"while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/splitЫ
)while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_82/MatMul_1/ReadVariableOpМ
while/gru_cell_82/MatMul_1MatMulwhile_placeholder_21while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMul_1С
while/gru_cell_82/BiasAdd_1BiasAdd$while/gru_cell_82/MatMul_1:product:0"while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAdd_1
while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_82/Const_1
#while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_82/split_1/split_dim­
while/gru_cell_82/split_1SplitV$while/gru_cell_82/BiasAdd_1:output:0"while/gru_cell_82/Const_1:output:0,while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/split_1Џ
while/gru_cell_82/addAddV2 while/gru_cell_82/split:output:0"while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add
while/gru_cell_82/SigmoidSigmoidwhile/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/SigmoidГ
while/gru_cell_82/add_1AddV2 while/gru_cell_82/split:output:1"while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_1
while/gru_cell_82/Sigmoid_1Sigmoidwhile/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Sigmoid_1Ќ
while/gru_cell_82/mulMulwhile/gru_cell_82/Sigmoid_1:y:0"while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mulЊ
while/gru_cell_82/add_2AddV2 while/gru_cell_82/split:output:2while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_2
while/gru_cell_82/TanhTanhwhile/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Tanh
while/gru_cell_82/mul_1Mulwhile/gru_cell_82/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_1w
while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_82/sub/xЈ
while/gru_cell_82/subSub while/gru_cell_82/sub/x:output:0while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/subЂ
while/gru_cell_82/mul_2Mulwhile/gru_cell_82/sub:z:0while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_2Ї
while/gru_cell_82/add_3AddV2while/gru_cell_82/mul_1:z:0while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_82/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_82_matmul_1_readvariableop_resource4while_gru_cell_82_matmul_1_readvariableop_resource_0"f
0while_gru_cell_82_matmul_readvariableop_resource2while_gru_cell_82_matmul_readvariableop_resource_0"X
)while_gru_cell_82_readvariableop_resource+while_gru_cell_82_readvariableop_resource_0")
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
Ћ
@__inference_OUT_layer_call_and_return_conditional_losses_1686536

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
оH
и
GRU_2_while_body_1687636(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_83_readvariableop_resource_0<
8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_83_readvariableop_resource:
6gru_2_while_gru_cell_83_matmul_readvariableop_resource<
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_83/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_83/ReadVariableOpВ
GRU_2/while/gru_cell_83/unstackUnpack.GRU_2/while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_83/unstackз
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_83/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_83/MatMulг
GRU_2/while/gru_cell_83/BiasAddBiasAdd(GRU_2/while/gru_cell_83/MatMul:product:0(GRU_2/while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_83/BiasAdd
GRU_2/while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_83/Const
'GRU_2/while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_83/split/split_dim
GRU_2/while/gru_cell_83/splitSplit0GRU_2/while/gru_cell_83/split/split_dim:output:0(GRU_2/while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_83/splitн
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_83/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_83/MatMul_1й
!GRU_2/while/gru_cell_83/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_83/MatMul_1:product:0(GRU_2/while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_83/BiasAdd_1
GRU_2/while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_83/Const_1Ё
)GRU_2/while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_83/split_1/split_dimЫ
GRU_2/while/gru_cell_83/split_1SplitV*GRU_2/while/gru_cell_83/BiasAdd_1:output:0(GRU_2/while/gru_cell_83/Const_1:output:02GRU_2/while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_83/split_1Ч
GRU_2/while/gru_cell_83/addAddV2&GRU_2/while/gru_cell_83/split:output:0(GRU_2/while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add 
GRU_2/while/gru_cell_83/SigmoidSigmoidGRU_2/while/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_83/SigmoidЫ
GRU_2/while/gru_cell_83/add_1AddV2&GRU_2/while/gru_cell_83/split:output:1(GRU_2/while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_1І
!GRU_2/while/gru_cell_83/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_83/Sigmoid_1Ф
GRU_2/while/gru_cell_83/mulMul%GRU_2/while/gru_cell_83/Sigmoid_1:y:0(GRU_2/while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mulТ
GRU_2/while/gru_cell_83/add_2AddV2&GRU_2/while/gru_cell_83/split:output:2GRU_2/while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_2
GRU_2/while/gru_cell_83/TanhTanh!GRU_2/while/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/TanhЗ
GRU_2/while/gru_cell_83/mul_1Mul#GRU_2/while/gru_cell_83/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_1
GRU_2/while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_83/sub/xР
GRU_2/while/gru_cell_83/subSub&GRU_2/while/gru_cell_83/sub/x:output:0#GRU_2/while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/subК
GRU_2/while/gru_cell_83/mul_2MulGRU_2/while/gru_cell_83/sub:z:0 GRU_2/while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_2П
GRU_2/while/gru_cell_83/add_3AddV2!GRU_2/while/gru_cell_83/mul_1:z:0!GRU_2/while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_83/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resource:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_83_matmul_readvariableop_resource8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_83_readvariableop_resource1gru_2_while_gru_cell_83_readvariableop_resource_0"5
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
е
Џ
while_cond_1688884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1688884___redundant_placeholder05
1while_while_cond_1688884___redundant_placeholder15
1while_while_cond_1688884___redundant_placeholder25
1while_while_cond_1688884___redundant_placeholder3
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
	
Ё
GRU_1_while_cond_1687097(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1687097___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1687097___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1687097___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1687097___redundant_placeholder3
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
ус

G__inference_Supervisor_layer_call_and_return_conditional_losses_1687370
gru_1_input-
)gru_1_gru_cell_82_readvariableop_resource4
0gru_1_gru_cell_82_matmul_readvariableop_resource6
2gru_1_gru_cell_82_matmul_1_readvariableop_resource-
)gru_2_gru_cell_83_readvariableop_resource4
0gru_2_gru_cell_83_matmul_readvariableop_resource6
2gru_2_gru_cell_83_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_82/ReadVariableOpReadVariableOp)gru_1_gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_82/ReadVariableOp 
GRU_1/gru_cell_82/unstackUnpack(GRU_1/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_82/unstackУ
'GRU_1/gru_cell_82/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_82/MatMul/ReadVariableOpС
GRU_1/gru_cell_82/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMulЛ
GRU_1/gru_cell_82/BiasAddBiasAdd"GRU_1/gru_cell_82/MatMul:product:0"GRU_1/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAddt
GRU_1/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_82/Const
!GRU_1/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_82/split/split_dimє
GRU_1/gru_cell_82/splitSplit*GRU_1/gru_cell_82/split/split_dim:output:0"GRU_1/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/splitЩ
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_82/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMul_1С
GRU_1/gru_cell_82/BiasAdd_1BiasAdd$GRU_1/gru_cell_82/MatMul_1:product:0"GRU_1/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAdd_1
GRU_1/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_82/Const_1
#GRU_1/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_82/split_1/split_dim­
GRU_1/gru_cell_82/split_1SplitV$GRU_1/gru_cell_82/BiasAdd_1:output:0"GRU_1/gru_cell_82/Const_1:output:0,GRU_1/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/split_1Џ
GRU_1/gru_cell_82/addAddV2 GRU_1/gru_cell_82/split:output:0"GRU_1/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add
GRU_1/gru_cell_82/SigmoidSigmoidGRU_1/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/SigmoidГ
GRU_1/gru_cell_82/add_1AddV2 GRU_1/gru_cell_82/split:output:1"GRU_1/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_1
GRU_1/gru_cell_82/Sigmoid_1SigmoidGRU_1/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Sigmoid_1Ќ
GRU_1/gru_cell_82/mulMulGRU_1/gru_cell_82/Sigmoid_1:y:0"GRU_1/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mulЊ
GRU_1/gru_cell_82/add_2AddV2 GRU_1/gru_cell_82/split:output:2GRU_1/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_2
GRU_1/gru_cell_82/TanhTanhGRU_1/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Tanh 
GRU_1/gru_cell_82/mul_1MulGRU_1/gru_cell_82/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_1w
GRU_1/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_82/sub/xЈ
GRU_1/gru_cell_82/subSub GRU_1/gru_cell_82/sub/x:output:0GRU_1/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/subЂ
GRU_1/gru_cell_82/mul_2MulGRU_1/gru_cell_82/sub:z:0GRU_1/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_2Ї
GRU_1/gru_cell_82/add_3AddV2GRU_1/gru_cell_82/mul_1:z:0GRU_1/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_3
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
GRU_1/while/loop_counter
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_82_readvariableop_resource0gru_1_gru_cell_82_matmul_readvariableop_resource2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_1_while_body_1687098*$
condR
GRU_1_while_cond_1687097*8
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
 GRU_2/gru_cell_83/ReadVariableOpReadVariableOp)gru_2_gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_83/ReadVariableOp 
GRU_2/gru_cell_83/unstackUnpack(GRU_2/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_83/unstackУ
'GRU_2/gru_cell_83/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_83/MatMul/ReadVariableOpС
GRU_2/gru_cell_83/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMulЛ
GRU_2/gru_cell_83/BiasAddBiasAdd"GRU_2/gru_cell_83/MatMul:product:0"GRU_2/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAddt
GRU_2/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_83/Const
!GRU_2/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_83/split/split_dimє
GRU_2/gru_cell_83/splitSplit*GRU_2/gru_cell_83/split/split_dim:output:0"GRU_2/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/splitЩ
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_83/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMul_1С
GRU_2/gru_cell_83/BiasAdd_1BiasAdd$GRU_2/gru_cell_83/MatMul_1:product:0"GRU_2/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAdd_1
GRU_2/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_83/Const_1
#GRU_2/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_83/split_1/split_dim­
GRU_2/gru_cell_83/split_1SplitV$GRU_2/gru_cell_83/BiasAdd_1:output:0"GRU_2/gru_cell_83/Const_1:output:0,GRU_2/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/split_1Џ
GRU_2/gru_cell_83/addAddV2 GRU_2/gru_cell_83/split:output:0"GRU_2/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add
GRU_2/gru_cell_83/SigmoidSigmoidGRU_2/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/SigmoidГ
GRU_2/gru_cell_83/add_1AddV2 GRU_2/gru_cell_83/split:output:1"GRU_2/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_1
GRU_2/gru_cell_83/Sigmoid_1SigmoidGRU_2/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Sigmoid_1Ќ
GRU_2/gru_cell_83/mulMulGRU_2/gru_cell_83/Sigmoid_1:y:0"GRU_2/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mulЊ
GRU_2/gru_cell_83/add_2AddV2 GRU_2/gru_cell_83/split:output:2GRU_2/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_2
GRU_2/gru_cell_83/TanhTanhGRU_2/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Tanh 
GRU_2/gru_cell_83/mul_1MulGRU_2/gru_cell_83/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_1w
GRU_2/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_83/sub/xЈ
GRU_2/gru_cell_83/subSub GRU_2/gru_cell_83/sub/x:output:0GRU_2/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/subЂ
GRU_2/gru_cell_83/mul_2MulGRU_2/gru_cell_83/sub:z:0GRU_2/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_2Ї
GRU_2/gru_cell_83/add_3AddV2GRU_2/gru_cell_83/mul_1:z:0GRU_2/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_3
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
GRU_2/while/loop_counter
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_83_readvariableop_resource0gru_2_gru_cell_83_matmul_readvariableop_resource2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_2_while_body_1687253*$
condR
GRU_2_while_cond_1687252*8
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
е
Џ
while_cond_1689383
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1689383___redundant_placeholder05
1while_while_cond_1689383___redundant_placeholder15
1while_while_cond_1689383___redundant_placeholder25
1while_while_cond_1689383___redundant_placeholder3
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
Т
р
,__inference_Supervisor_layer_call_fn_1687391
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЮ
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
GPU 2J 8 *P
fKRI
G__inference_Supervisor_layer_call_and_return_conditional_losses_16866022
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
Яс

G__inference_Supervisor_layer_call_and_return_conditional_losses_1688094

inputs-
)gru_1_gru_cell_82_readvariableop_resource4
0gru_1_gru_cell_82_matmul_readvariableop_resource6
2gru_1_gru_cell_82_matmul_1_readvariableop_resource-
)gru_2_gru_cell_83_readvariableop_resource4
0gru_2_gru_cell_83_matmul_readvariableop_resource6
2gru_2_gru_cell_83_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_82/ReadVariableOpReadVariableOp)gru_1_gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_82/ReadVariableOp 
GRU_1/gru_cell_82/unstackUnpack(GRU_1/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_82/unstackУ
'GRU_1/gru_cell_82/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_82/MatMul/ReadVariableOpС
GRU_1/gru_cell_82/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMulЛ
GRU_1/gru_cell_82/BiasAddBiasAdd"GRU_1/gru_cell_82/MatMul:product:0"GRU_1/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAddt
GRU_1/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_82/Const
!GRU_1/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_82/split/split_dimє
GRU_1/gru_cell_82/splitSplit*GRU_1/gru_cell_82/split/split_dim:output:0"GRU_1/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/splitЩ
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_82/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMul_1С
GRU_1/gru_cell_82/BiasAdd_1BiasAdd$GRU_1/gru_cell_82/MatMul_1:product:0"GRU_1/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAdd_1
GRU_1/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_82/Const_1
#GRU_1/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_82/split_1/split_dim­
GRU_1/gru_cell_82/split_1SplitV$GRU_1/gru_cell_82/BiasAdd_1:output:0"GRU_1/gru_cell_82/Const_1:output:0,GRU_1/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/split_1Џ
GRU_1/gru_cell_82/addAddV2 GRU_1/gru_cell_82/split:output:0"GRU_1/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add
GRU_1/gru_cell_82/SigmoidSigmoidGRU_1/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/SigmoidГ
GRU_1/gru_cell_82/add_1AddV2 GRU_1/gru_cell_82/split:output:1"GRU_1/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_1
GRU_1/gru_cell_82/Sigmoid_1SigmoidGRU_1/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Sigmoid_1Ќ
GRU_1/gru_cell_82/mulMulGRU_1/gru_cell_82/Sigmoid_1:y:0"GRU_1/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mulЊ
GRU_1/gru_cell_82/add_2AddV2 GRU_1/gru_cell_82/split:output:2GRU_1/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_2
GRU_1/gru_cell_82/TanhTanhGRU_1/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Tanh 
GRU_1/gru_cell_82/mul_1MulGRU_1/gru_cell_82/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_1w
GRU_1/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_82/sub/xЈ
GRU_1/gru_cell_82/subSub GRU_1/gru_cell_82/sub/x:output:0GRU_1/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/subЂ
GRU_1/gru_cell_82/mul_2MulGRU_1/gru_cell_82/sub:z:0GRU_1/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_2Ї
GRU_1/gru_cell_82/add_3AddV2GRU_1/gru_cell_82/mul_1:z:0GRU_1/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_3
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
GRU_1/while/loop_counter
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_82_readvariableop_resource0gru_1_gru_cell_82_matmul_readvariableop_resource2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_1_while_body_1687822*$
condR
GRU_1_while_cond_1687821*8
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
 GRU_2/gru_cell_83/ReadVariableOpReadVariableOp)gru_2_gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_83/ReadVariableOp 
GRU_2/gru_cell_83/unstackUnpack(GRU_2/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_83/unstackУ
'GRU_2/gru_cell_83/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_83/MatMul/ReadVariableOpС
GRU_2/gru_cell_83/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMulЛ
GRU_2/gru_cell_83/BiasAddBiasAdd"GRU_2/gru_cell_83/MatMul:product:0"GRU_2/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAddt
GRU_2/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_83/Const
!GRU_2/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_83/split/split_dimє
GRU_2/gru_cell_83/splitSplit*GRU_2/gru_cell_83/split/split_dim:output:0"GRU_2/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/splitЩ
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_83/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMul_1С
GRU_2/gru_cell_83/BiasAdd_1BiasAdd$GRU_2/gru_cell_83/MatMul_1:product:0"GRU_2/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAdd_1
GRU_2/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_83/Const_1
#GRU_2/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_83/split_1/split_dim­
GRU_2/gru_cell_83/split_1SplitV$GRU_2/gru_cell_83/BiasAdd_1:output:0"GRU_2/gru_cell_83/Const_1:output:0,GRU_2/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/split_1Џ
GRU_2/gru_cell_83/addAddV2 GRU_2/gru_cell_83/split:output:0"GRU_2/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add
GRU_2/gru_cell_83/SigmoidSigmoidGRU_2/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/SigmoidГ
GRU_2/gru_cell_83/add_1AddV2 GRU_2/gru_cell_83/split:output:1"GRU_2/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_1
GRU_2/gru_cell_83/Sigmoid_1SigmoidGRU_2/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Sigmoid_1Ќ
GRU_2/gru_cell_83/mulMulGRU_2/gru_cell_83/Sigmoid_1:y:0"GRU_2/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mulЊ
GRU_2/gru_cell_83/add_2AddV2 GRU_2/gru_cell_83/split:output:2GRU_2/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_2
GRU_2/gru_cell_83/TanhTanhGRU_2/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Tanh 
GRU_2/gru_cell_83/mul_1MulGRU_2/gru_cell_83/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_1w
GRU_2/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_83/sub/xЈ
GRU_2/gru_cell_83/subSub GRU_2/gru_cell_83/sub/x:output:0GRU_2/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/subЂ
GRU_2/gru_cell_83/mul_2MulGRU_2/gru_cell_83/sub:z:0GRU_2/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_2Ї
GRU_2/gru_cell_83/add_3AddV2GRU_2/gru_cell_83/mul_1:z:0GRU_2/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_3
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
GRU_2/while/loop_counter
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_83_readvariableop_resource0gru_2_gru_cell_83_matmul_readvariableop_resource2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_2_while_body_1687977*$
condR
GRU_2_while_cond_1687976*8
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
ЊX
і
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689474
inputs_0'
#gru_cell_83_readvariableop_resource.
*gru_cell_83_matmul_readvariableop_resource0
,gru_cell_83_matmul_1_readvariableop_resource
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
gru_cell_83/ReadVariableOpReadVariableOp#gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_83/ReadVariableOp
gru_cell_83/unstackUnpack"gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_83/unstackБ
!gru_cell_83/MatMul/ReadVariableOpReadVariableOp*gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_83/MatMul/ReadVariableOpЉ
gru_cell_83/MatMulMatMulstrided_slice_2:output:0)gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMulЃ
gru_cell_83/BiasAddBiasAddgru_cell_83/MatMul:product:0gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAddh
gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_83/Const
gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split/split_dimм
gru_cell_83/splitSplit$gru_cell_83/split/split_dim:output:0gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/splitЗ
#gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_83/MatMul_1/ReadVariableOpЅ
gru_cell_83/MatMul_1MatMulzeros:output:0+gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMul_1Љ
gru_cell_83/BiasAdd_1BiasAddgru_cell_83/MatMul_1:product:0gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAdd_1
gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_83/Const_1
gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split_1/split_dim
gru_cell_83/split_1SplitVgru_cell_83/BiasAdd_1:output:0gru_cell_83/Const_1:output:0&gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/split_1
gru_cell_83/addAddV2gru_cell_83/split:output:0gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add|
gru_cell_83/SigmoidSigmoidgru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid
gru_cell_83/add_1AddV2gru_cell_83/split:output:1gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_1
gru_cell_83/Sigmoid_1Sigmoidgru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid_1
gru_cell_83/mulMulgru_cell_83/Sigmoid_1:y:0gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul
gru_cell_83/add_2AddV2gru_cell_83/split:output:2gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_2u
gru_cell_83/TanhTanhgru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Tanh
gru_cell_83/mul_1Mulgru_cell_83/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_1k
gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_83/sub/x
gru_cell_83/subSubgru_cell_83/sub/x:output:0gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/sub
gru_cell_83/mul_2Mulgru_cell_83/sub:z:0gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_2
gru_cell_83/add_3AddV2gru_cell_83/mul_1:z:0gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_83_readvariableop_resource*gru_cell_83_matmul_readvariableop_resource,gru_cell_83_matmul_1_readvariableop_resource*
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
bodyR
while_body_1689384*
condR
while_cond_1689383*8
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
Г

'__inference_GRU_2_layer_call_fn_1689485
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_2_layer_call_and_return_conditional_losses_16856802
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
@
Ж
while_body_1686385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_83_readvariableop_resource_06
2while_gru_cell_83_matmul_readvariableop_resource_08
4while_gru_cell_83_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_83_readvariableop_resource4
0while_gru_cell_83_matmul_readvariableop_resource6
2while_gru_cell_83_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_83/ReadVariableOpReadVariableOp+while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_83/ReadVariableOp 
while/gru_cell_83/unstackUnpack(while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_83/unstackХ
'while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_83/MatMul/ReadVariableOpг
while/gru_cell_83/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMulЛ
while/gru_cell_83/BiasAddBiasAdd"while/gru_cell_83/MatMul:product:0"while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAddt
while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_83/Const
!while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_83/split/split_dimє
while/gru_cell_83/splitSplit*while/gru_cell_83/split/split_dim:output:0"while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/splitЫ
)while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_83/MatMul_1/ReadVariableOpМ
while/gru_cell_83/MatMul_1MatMulwhile_placeholder_21while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMul_1С
while/gru_cell_83/BiasAdd_1BiasAdd$while/gru_cell_83/MatMul_1:product:0"while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAdd_1
while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_83/Const_1
#while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_83/split_1/split_dim­
while/gru_cell_83/split_1SplitV$while/gru_cell_83/BiasAdd_1:output:0"while/gru_cell_83/Const_1:output:0,while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/split_1Џ
while/gru_cell_83/addAddV2 while/gru_cell_83/split:output:0"while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add
while/gru_cell_83/SigmoidSigmoidwhile/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/SigmoidГ
while/gru_cell_83/add_1AddV2 while/gru_cell_83/split:output:1"while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_1
while/gru_cell_83/Sigmoid_1Sigmoidwhile/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Sigmoid_1Ќ
while/gru_cell_83/mulMulwhile/gru_cell_83/Sigmoid_1:y:0"while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mulЊ
while/gru_cell_83/add_2AddV2 while/gru_cell_83/split:output:2while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_2
while/gru_cell_83/TanhTanhwhile/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Tanh
while/gru_cell_83/mul_1Mulwhile/gru_cell_83/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_1w
while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_83/sub/xЈ
while/gru_cell_83/subSub while/gru_cell_83/sub/x:output:0while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/subЂ
while/gru_cell_83/mul_2Mulwhile/gru_cell_83/sub:z:0while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_2Ї
while/gru_cell_83/add_3AddV2while/gru_cell_83/mul_1:z:0while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_83/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_83_matmul_1_readvariableop_resource4while_gru_cell_83_matmul_1_readvariableop_resource_0"f
0while_gru_cell_83_matmul_readvariableop_resource2while_gru_cell_83_matmul_readvariableop_resource_0"X
)while_gru_cell_83_readvariableop_resource+while_gru_cell_83_readvariableop_resource_0")
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
@
Ж
while_body_1688885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_83_readvariableop_resource_06
2while_gru_cell_83_matmul_readvariableop_resource_08
4while_gru_cell_83_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_83_readvariableop_resource4
0while_gru_cell_83_matmul_readvariableop_resource6
2while_gru_cell_83_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_83/ReadVariableOpReadVariableOp+while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_83/ReadVariableOp 
while/gru_cell_83/unstackUnpack(while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_83/unstackХ
'while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_83/MatMul/ReadVariableOpг
while/gru_cell_83/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMulЛ
while/gru_cell_83/BiasAddBiasAdd"while/gru_cell_83/MatMul:product:0"while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAddt
while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_83/Const
!while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_83/split/split_dimє
while/gru_cell_83/splitSplit*while/gru_cell_83/split/split_dim:output:0"while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/splitЫ
)while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_83/MatMul_1/ReadVariableOpМ
while/gru_cell_83/MatMul_1MatMulwhile_placeholder_21while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMul_1С
while/gru_cell_83/BiasAdd_1BiasAdd$while/gru_cell_83/MatMul_1:product:0"while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAdd_1
while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_83/Const_1
#while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_83/split_1/split_dim­
while/gru_cell_83/split_1SplitV$while/gru_cell_83/BiasAdd_1:output:0"while/gru_cell_83/Const_1:output:0,while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/split_1Џ
while/gru_cell_83/addAddV2 while/gru_cell_83/split:output:0"while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add
while/gru_cell_83/SigmoidSigmoidwhile/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/SigmoidГ
while/gru_cell_83/add_1AddV2 while/gru_cell_83/split:output:1"while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_1
while/gru_cell_83/Sigmoid_1Sigmoidwhile/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Sigmoid_1Ќ
while/gru_cell_83/mulMulwhile/gru_cell_83/Sigmoid_1:y:0"while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mulЊ
while/gru_cell_83/add_2AddV2 while/gru_cell_83/split:output:2while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_2
while/gru_cell_83/TanhTanhwhile/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Tanh
while/gru_cell_83/mul_1Mulwhile/gru_cell_83/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_1w
while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_83/sub/xЈ
while/gru_cell_83/subSub while/gru_cell_83/sub/x:output:0while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/subЂ
while/gru_cell_83/mul_2Mulwhile/gru_cell_83/sub:z:0while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_2Ї
while/gru_cell_83/add_3AddV2while/gru_cell_83/mul_1:z:0while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_83/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_83_matmul_1_readvariableop_resource4while_gru_cell_83_matmul_1_readvariableop_resource_0"f
0while_gru_cell_83_matmul_readvariableop_resource2while_gru_cell_83_matmul_readvariableop_resource_0"X
)while_gru_cell_83_readvariableop_resource+while_gru_cell_83_readvariableop_resource_0")
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


'__inference_GRU_2_layer_call_fn_1689156

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_2_layer_call_and_return_conditional_losses_16864752
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
ч
ъ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1685357

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
оH
и
GRU_2_while_body_1687977(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_83_readvariableop_resource_0<
8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_83_readvariableop_resource:
6gru_2_while_gru_cell_83_matmul_readvariableop_resource<
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_83/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_83/ReadVariableOpВ
GRU_2/while/gru_cell_83/unstackUnpack.GRU_2/while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_83/unstackз
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_83/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_83/MatMulг
GRU_2/while/gru_cell_83/BiasAddBiasAdd(GRU_2/while/gru_cell_83/MatMul:product:0(GRU_2/while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_83/BiasAdd
GRU_2/while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_83/Const
'GRU_2/while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_83/split/split_dim
GRU_2/while/gru_cell_83/splitSplit0GRU_2/while/gru_cell_83/split/split_dim:output:0(GRU_2/while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_83/splitн
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_83/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_83/MatMul_1й
!GRU_2/while/gru_cell_83/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_83/MatMul_1:product:0(GRU_2/while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_83/BiasAdd_1
GRU_2/while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_83/Const_1Ё
)GRU_2/while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_83/split_1/split_dimЫ
GRU_2/while/gru_cell_83/split_1SplitV*GRU_2/while/gru_cell_83/BiasAdd_1:output:0(GRU_2/while/gru_cell_83/Const_1:output:02GRU_2/while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_83/split_1Ч
GRU_2/while/gru_cell_83/addAddV2&GRU_2/while/gru_cell_83/split:output:0(GRU_2/while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add 
GRU_2/while/gru_cell_83/SigmoidSigmoidGRU_2/while/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_83/SigmoidЫ
GRU_2/while/gru_cell_83/add_1AddV2&GRU_2/while/gru_cell_83/split:output:1(GRU_2/while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_1І
!GRU_2/while/gru_cell_83/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_83/Sigmoid_1Ф
GRU_2/while/gru_cell_83/mulMul%GRU_2/while/gru_cell_83/Sigmoid_1:y:0(GRU_2/while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mulТ
GRU_2/while/gru_cell_83/add_2AddV2&GRU_2/while/gru_cell_83/split:output:2GRU_2/while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_2
GRU_2/while/gru_cell_83/TanhTanh!GRU_2/while/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/TanhЗ
GRU_2/while/gru_cell_83/mul_1Mul#GRU_2/while/gru_cell_83/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_1
GRU_2/while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_83/sub/xР
GRU_2/while/gru_cell_83/subSub&GRU_2/while/gru_cell_83/sub/x:output:0#GRU_2/while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/subК
GRU_2/while/gru_cell_83/mul_2MulGRU_2/while/gru_cell_83/sub:z:0 GRU_2/while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_2П
GRU_2/while/gru_cell_83/add_3AddV2!GRU_2/while/gru_cell_83/mul_1:z:0!GRU_2/while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_83/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resource:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_83_matmul_readvariableop_resource8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_83_readvariableop_resource1gru_2_while_gru_cell_83_readvariableop_resource_0"5
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
џ!
т
while_body_1685734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_83_1685756_0
while_gru_cell_83_1685758_0
while_gru_cell_83_1685760_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_83_1685756
while_gru_cell_83_1685758
while_gru_cell_83_1685760Ђ)while/gru_cell_83/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemЕ
)while/gru_cell_83/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_83_1685756_0while_gru_cell_83_1685758_0while_gru_cell_83_1685760_0*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_16853572+
)while/gru_cell_83/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_83/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_83/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_83/StatefulPartitionedCall:output:1*^while/gru_cell_83/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_83_1685756while_gru_cell_83_1685756_0"8
while_gru_cell_83_1685758while_gru_cell_83_1685758_0"8
while_gru_cell_83_1685760while_gru_cell_83_1685760_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_83/StatefulPartitionedCall)while/gru_cell_83/StatefulPartitionedCall: 
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
ч
ъ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1685317

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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688794

inputs'
#gru_cell_82_readvariableop_resource.
*gru_cell_82_matmul_readvariableop_resource0
,gru_cell_82_matmul_1_readvariableop_resource
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
gru_cell_82/ReadVariableOpReadVariableOp#gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_82/ReadVariableOp
gru_cell_82/unstackUnpack"gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_82/unstackБ
!gru_cell_82/MatMul/ReadVariableOpReadVariableOp*gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_82/MatMul/ReadVariableOpЉ
gru_cell_82/MatMulMatMulstrided_slice_2:output:0)gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMulЃ
gru_cell_82/BiasAddBiasAddgru_cell_82/MatMul:product:0gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAddh
gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_82/Const
gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split/split_dimм
gru_cell_82/splitSplit$gru_cell_82/split/split_dim:output:0gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/splitЗ
#gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_82/MatMul_1/ReadVariableOpЅ
gru_cell_82/MatMul_1MatMulzeros:output:0+gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMul_1Љ
gru_cell_82/BiasAdd_1BiasAddgru_cell_82/MatMul_1:product:0gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAdd_1
gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_82/Const_1
gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split_1/split_dim
gru_cell_82/split_1SplitVgru_cell_82/BiasAdd_1:output:0gru_cell_82/Const_1:output:0&gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/split_1
gru_cell_82/addAddV2gru_cell_82/split:output:0gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add|
gru_cell_82/SigmoidSigmoidgru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid
gru_cell_82/add_1AddV2gru_cell_82/split:output:1gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_1
gru_cell_82/Sigmoid_1Sigmoidgru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid_1
gru_cell_82/mulMulgru_cell_82/Sigmoid_1:y:0gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul
gru_cell_82/add_2AddV2gru_cell_82/split:output:2gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_2u
gru_cell_82/TanhTanhgru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Tanh
gru_cell_82/mul_1Mulgru_cell_82/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_1k
gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_82/sub/x
gru_cell_82/subSubgru_cell_82/sub/x:output:0gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/sub
gru_cell_82/mul_2Mulgru_cell_82/sub:z:0gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_2
gru_cell_82/add_3AddV2gru_cell_82/mul_1:z:0gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_82_readvariableop_resource*gru_cell_82_matmul_readvariableop_resource,gru_cell_82_matmul_1_readvariableop_resource*
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
bodyR
while_body_1688704*
condR
while_cond_1688703*8
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
@
Ж
while_body_1688704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_82_readvariableop_resource_06
2while_gru_cell_82_matmul_readvariableop_resource_08
4while_gru_cell_82_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_82_readvariableop_resource4
0while_gru_cell_82_matmul_readvariableop_resource6
2while_gru_cell_82_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_82/ReadVariableOpReadVariableOp+while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_82/ReadVariableOp 
while/gru_cell_82/unstackUnpack(while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_82/unstackХ
'while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_82/MatMul/ReadVariableOpг
while/gru_cell_82/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMulЛ
while/gru_cell_82/BiasAddBiasAdd"while/gru_cell_82/MatMul:product:0"while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAddt
while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_82/Const
!while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_82/split/split_dimє
while/gru_cell_82/splitSplit*while/gru_cell_82/split/split_dim:output:0"while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/splitЫ
)while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_82/MatMul_1/ReadVariableOpМ
while/gru_cell_82/MatMul_1MatMulwhile_placeholder_21while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMul_1С
while/gru_cell_82/BiasAdd_1BiasAdd$while/gru_cell_82/MatMul_1:product:0"while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAdd_1
while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_82/Const_1
#while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_82/split_1/split_dim­
while/gru_cell_82/split_1SplitV$while/gru_cell_82/BiasAdd_1:output:0"while/gru_cell_82/Const_1:output:0,while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/split_1Џ
while/gru_cell_82/addAddV2 while/gru_cell_82/split:output:0"while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add
while/gru_cell_82/SigmoidSigmoidwhile/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/SigmoidГ
while/gru_cell_82/add_1AddV2 while/gru_cell_82/split:output:1"while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_1
while/gru_cell_82/Sigmoid_1Sigmoidwhile/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Sigmoid_1Ќ
while/gru_cell_82/mulMulwhile/gru_cell_82/Sigmoid_1:y:0"while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mulЊ
while/gru_cell_82/add_2AddV2 while/gru_cell_82/split:output:2while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_2
while/gru_cell_82/TanhTanhwhile/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Tanh
while/gru_cell_82/mul_1Mulwhile/gru_cell_82/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_1w
while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_82/sub/xЈ
while/gru_cell_82/subSub while/gru_cell_82/sub/x:output:0while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/subЂ
while/gru_cell_82/mul_2Mulwhile/gru_cell_82/sub:z:0while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_2Ї
while/gru_cell_82/add_3AddV2while/gru_cell_82/mul_1:z:0while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_82/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_82_matmul_1_readvariableop_resource4while_gru_cell_82_matmul_1_readvariableop_resource_0"f
0while_gru_cell_82_matmul_readvariableop_resource2while_gru_cell_82_matmul_readvariableop_resource_0"X
)while_gru_cell_82_readvariableop_resource+while_gru_cell_82_readvariableop_resource_0")
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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1685969

inputs'
#gru_cell_82_readvariableop_resource.
*gru_cell_82_matmul_readvariableop_resource0
,gru_cell_82_matmul_1_readvariableop_resource
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
gru_cell_82/ReadVariableOpReadVariableOp#gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_82/ReadVariableOp
gru_cell_82/unstackUnpack"gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_82/unstackБ
!gru_cell_82/MatMul/ReadVariableOpReadVariableOp*gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_82/MatMul/ReadVariableOpЉ
gru_cell_82/MatMulMatMulstrided_slice_2:output:0)gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMulЃ
gru_cell_82/BiasAddBiasAddgru_cell_82/MatMul:product:0gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAddh
gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_82/Const
gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split/split_dimм
gru_cell_82/splitSplit$gru_cell_82/split/split_dim:output:0gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/splitЗ
#gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_82/MatMul_1/ReadVariableOpЅ
gru_cell_82/MatMul_1MatMulzeros:output:0+gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMul_1Љ
gru_cell_82/BiasAdd_1BiasAddgru_cell_82/MatMul_1:product:0gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAdd_1
gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_82/Const_1
gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split_1/split_dim
gru_cell_82/split_1SplitVgru_cell_82/BiasAdd_1:output:0gru_cell_82/Const_1:output:0&gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/split_1
gru_cell_82/addAddV2gru_cell_82/split:output:0gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add|
gru_cell_82/SigmoidSigmoidgru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid
gru_cell_82/add_1AddV2gru_cell_82/split:output:1gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_1
gru_cell_82/Sigmoid_1Sigmoidgru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid_1
gru_cell_82/mulMulgru_cell_82/Sigmoid_1:y:0gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul
gru_cell_82/add_2AddV2gru_cell_82/split:output:2gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_2u
gru_cell_82/TanhTanhgru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Tanh
gru_cell_82/mul_1Mulgru_cell_82/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_1k
gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_82/sub/x
gru_cell_82/subSubgru_cell_82/sub/x:output:0gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/sub
gru_cell_82/mul_2Mulgru_cell_82/sub:z:0gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_2
gru_cell_82/add_3AddV2gru_cell_82/mul_1:z:0gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_82_readvariableop_resource*gru_cell_82_matmul_readvariableop_resource,gru_cell_82_matmul_1_readvariableop_resource*
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
bodyR
while_body_1685879*
condR
while_cond_1685878*8
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
Г
л
,__inference_Supervisor_layer_call_fn_1688136

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЩ
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
GPU 2J 8 *P
fKRI
G__inference_Supervisor_layer_call_and_return_conditional_losses_16866462
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
Г
л
,__inference_Supervisor_layer_call_fn_1688115

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЩ
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
GPU 2J 8 *P
fKRI
G__inference_Supervisor_layer_call_and_return_conditional_losses_16866022
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
ч
ъ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1684795

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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1686128

inputs'
#gru_cell_82_readvariableop_resource.
*gru_cell_82_matmul_readvariableop_resource0
,gru_cell_82_matmul_1_readvariableop_resource
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
gru_cell_82/ReadVariableOpReadVariableOp#gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_82/ReadVariableOp
gru_cell_82/unstackUnpack"gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_82/unstackБ
!gru_cell_82/MatMul/ReadVariableOpReadVariableOp*gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_82/MatMul/ReadVariableOpЉ
gru_cell_82/MatMulMatMulstrided_slice_2:output:0)gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMulЃ
gru_cell_82/BiasAddBiasAddgru_cell_82/MatMul:product:0gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAddh
gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_82/Const
gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split/split_dimм
gru_cell_82/splitSplit$gru_cell_82/split/split_dim:output:0gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/splitЗ
#gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_82/MatMul_1/ReadVariableOpЅ
gru_cell_82/MatMul_1MatMulzeros:output:0+gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMul_1Љ
gru_cell_82/BiasAdd_1BiasAddgru_cell_82/MatMul_1:product:0gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAdd_1
gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_82/Const_1
gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split_1/split_dim
gru_cell_82/split_1SplitVgru_cell_82/BiasAdd_1:output:0gru_cell_82/Const_1:output:0&gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/split_1
gru_cell_82/addAddV2gru_cell_82/split:output:0gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add|
gru_cell_82/SigmoidSigmoidgru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid
gru_cell_82/add_1AddV2gru_cell_82/split:output:1gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_1
gru_cell_82/Sigmoid_1Sigmoidgru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid_1
gru_cell_82/mulMulgru_cell_82/Sigmoid_1:y:0gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul
gru_cell_82/add_2AddV2gru_cell_82/split:output:2gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_2u
gru_cell_82/TanhTanhgru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Tanh
gru_cell_82/mul_1Mulgru_cell_82/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_1k
gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_82/sub/x
gru_cell_82/subSubgru_cell_82/sub/x:output:0gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/sub
gru_cell_82/mul_2Mulgru_cell_82/sub:z:0gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_2
gru_cell_82/add_3AddV2gru_cell_82/mul_1:z:0gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_82_readvariableop_resource*gru_cell_82_matmul_readvariableop_resource,gru_cell_82_matmul_1_readvariableop_resource*
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
bodyR
while_body_1686038*
condR
while_cond_1686037*8
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
	
Ё
GRU_1_while_cond_1687480(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1687480___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1687480___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1687480___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1687480___redundant_placeholder3
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
џ!
т
while_body_1685054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_82_1685076_0
while_gru_cell_82_1685078_0
while_gru_cell_82_1685080_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_82_1685076
while_gru_cell_82_1685078
while_gru_cell_82_1685080Ђ)while/gru_cell_82/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemЕ
)while/gru_cell_82/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_82_1685076_0while_gru_cell_82_1685078_0while_gru_cell_82_1685080_0*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_16847552+
)while/gru_cell_82/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_82/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_82/StatefulPartitionedCall:output:1*^while/gru_cell_82/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_82_1685076while_gru_cell_82_1685076_0"8
while_gru_cell_82_1685078while_gru_cell_82_1685078_0"8
while_gru_cell_82_1685080while_gru_cell_82_1685080_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_82/StatefulPartitionedCall)while/gru_cell_82/StatefulPartitionedCall: 
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
оH
и
GRU_2_while_body_1687253(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_83_readvariableop_resource_0<
8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_83_readvariableop_resource:
6gru_2_while_gru_cell_83_matmul_readvariableop_resource<
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_83/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_83/ReadVariableOpВ
GRU_2/while/gru_cell_83/unstackUnpack.GRU_2/while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_83/unstackз
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_83/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_83/MatMulг
GRU_2/while/gru_cell_83/BiasAddBiasAdd(GRU_2/while/gru_cell_83/MatMul:product:0(GRU_2/while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_83/BiasAdd
GRU_2/while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_83/Const
'GRU_2/while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_83/split/split_dim
GRU_2/while/gru_cell_83/splitSplit0GRU_2/while/gru_cell_83/split/split_dim:output:0(GRU_2/while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_83/splitн
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_83/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_83/MatMul_1й
!GRU_2/while/gru_cell_83/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_83/MatMul_1:product:0(GRU_2/while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_83/BiasAdd_1
GRU_2/while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_83/Const_1Ё
)GRU_2/while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_83/split_1/split_dimЫ
GRU_2/while/gru_cell_83/split_1SplitV*GRU_2/while/gru_cell_83/BiasAdd_1:output:0(GRU_2/while/gru_cell_83/Const_1:output:02GRU_2/while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_83/split_1Ч
GRU_2/while/gru_cell_83/addAddV2&GRU_2/while/gru_cell_83/split:output:0(GRU_2/while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add 
GRU_2/while/gru_cell_83/SigmoidSigmoidGRU_2/while/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_83/SigmoidЫ
GRU_2/while/gru_cell_83/add_1AddV2&GRU_2/while/gru_cell_83/split:output:1(GRU_2/while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_1І
!GRU_2/while/gru_cell_83/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_83/Sigmoid_1Ф
GRU_2/while/gru_cell_83/mulMul%GRU_2/while/gru_cell_83/Sigmoid_1:y:0(GRU_2/while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mulТ
GRU_2/while/gru_cell_83/add_2AddV2&GRU_2/while/gru_cell_83/split:output:2GRU_2/while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_2
GRU_2/while/gru_cell_83/TanhTanh!GRU_2/while/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/TanhЗ
GRU_2/while/gru_cell_83/mul_1Mul#GRU_2/while/gru_cell_83/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_1
GRU_2/while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_83/sub/xР
GRU_2/while/gru_cell_83/subSub&GRU_2/while/gru_cell_83/sub/x:output:0#GRU_2/while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/subК
GRU_2/while/gru_cell_83/mul_2MulGRU_2/while/gru_cell_83/sub:z:0 GRU_2/while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_2П
GRU_2/while/gru_cell_83/add_3AddV2!GRU_2/while/gru_cell_83/mul_1:z:0!GRU_2/while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_83/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resource:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_83_matmul_readvariableop_resource8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_83_readvariableop_resource1gru_2_while_gru_cell_83_readvariableop_resource_0"5
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
і<
к
B__inference_GRU_2_layer_call_and_return_conditional_losses_1685680

inputs
gru_cell_83_1685604
gru_cell_83_1685606
gru_cell_83_1685608
identityЂ#gru_cell_83/StatefulPartitionedCallЂwhileD
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
strided_slice_2є
#gru_cell_83/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_83_1685604gru_cell_83_1685606gru_cell_83_1685608*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_16853172%
#gru_cell_83/StatefulPartitionedCall
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
while/loop_counterь
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_83_1685604gru_cell_83_1685606gru_cell_83_1685608*
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
bodyR
while_body_1685616*
condR
while_cond_1685615*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_83/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_83/StatefulPartitionedCall#gru_cell_83/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ!
т
while_body_1685172
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_82_1685194_0
while_gru_cell_82_1685196_0
while_gru_cell_82_1685198_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_82_1685194
while_gru_cell_82_1685196
while_gru_cell_82_1685198Ђ)while/gru_cell_82/StatefulPartitionedCallУ
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
)while/TensorArrayV2Read/TensorListGetItemЕ
)while/gru_cell_82/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_82_1685194_0while_gru_cell_82_1685196_0while_gru_cell_82_1685198_0*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_16847952+
)while/gru_cell_82/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_82/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_82/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_82/StatefulPartitionedCall:output:1*^while/gru_cell_82/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_82_1685194while_gru_cell_82_1685194_0"8
while_gru_cell_82_1685196while_gru_cell_82_1685196_0"8
while_gru_cell_82_1685198while_gru_cell_82_1685198_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_82/StatefulPartitionedCall)while/gru_cell_82/StatefulPartitionedCall: 
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
ЊX


#Supervisor_GRU_2_while_body_1684566>
:supervisor_gru_2_while_supervisor_gru_2_while_loop_counterD
@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations&
"supervisor_gru_2_while_placeholder(
$supervisor_gru_2_while_placeholder_1(
$supervisor_gru_2_while_placeholder_2=
9supervisor_gru_2_while_supervisor_gru_2_strided_slice_1_0y
usupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor_0@
<supervisor_gru_2_while_gru_cell_83_readvariableop_resource_0G
Csupervisor_gru_2_while_gru_cell_83_matmul_readvariableop_resource_0I
Esupervisor_gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0#
supervisor_gru_2_while_identity%
!supervisor_gru_2_while_identity_1%
!supervisor_gru_2_while_identity_2%
!supervisor_gru_2_while_identity_3%
!supervisor_gru_2_while_identity_4;
7supervisor_gru_2_while_supervisor_gru_2_strided_slice_1w
ssupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor>
:supervisor_gru_2_while_gru_cell_83_readvariableop_resourceE
Asupervisor_gru_2_while_gru_cell_83_matmul_readvariableop_resourceG
Csupervisor_gru_2_while_gru_cell_83_matmul_1_readvariableop_resourceх
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
1Supervisor/GRU_2/while/gru_cell_83/ReadVariableOpReadVariableOp<supervisor_gru_2_while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype023
1Supervisor/GRU_2/while/gru_cell_83/ReadVariableOpг
*Supervisor/GRU_2/while/gru_cell_83/unstackUnpack9Supervisor/GRU_2/while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2,
*Supervisor/GRU_2/while/gru_cell_83/unstackј
8Supervisor/GRU_2/while/gru_cell_83/MatMul/ReadVariableOpReadVariableOpCsupervisor_gru_2_while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02:
8Supervisor/GRU_2/while/gru_cell_83/MatMul/ReadVariableOp
)Supervisor/GRU_2/while/gru_cell_83/MatMulMatMulASupervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:0@Supervisor/GRU_2/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2+
)Supervisor/GRU_2/while/gru_cell_83/MatMulџ
*Supervisor/GRU_2/while/gru_cell_83/BiasAddBiasAdd3Supervisor/GRU_2/while/gru_cell_83/MatMul:product:03Supervisor/GRU_2/while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2,
*Supervisor/GRU_2/while/gru_cell_83/BiasAdd
(Supervisor/GRU_2/while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(Supervisor/GRU_2/while/gru_cell_83/ConstГ
2Supervisor/GRU_2/while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2Supervisor/GRU_2/while/gru_cell_83/split/split_dimИ
(Supervisor/GRU_2/while/gru_cell_83/splitSplit;Supervisor/GRU_2/while/gru_cell_83/split/split_dim:output:03Supervisor/GRU_2/while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2*
(Supervisor/GRU_2/while/gru_cell_83/splitў
:Supervisor/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOpEsupervisor_gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02<
:Supervisor/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOp
+Supervisor/GRU_2/while/gru_cell_83/MatMul_1MatMul$supervisor_gru_2_while_placeholder_2BSupervisor/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2-
+Supervisor/GRU_2/while/gru_cell_83/MatMul_1
,Supervisor/GRU_2/while/gru_cell_83/BiasAdd_1BiasAdd5Supervisor/GRU_2/while/gru_cell_83/MatMul_1:product:03Supervisor/GRU_2/while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2.
,Supervisor/GRU_2/while/gru_cell_83/BiasAdd_1­
*Supervisor/GRU_2/while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2,
*Supervisor/GRU_2/while/gru_cell_83/Const_1З
4Supervisor/GRU_2/while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ26
4Supervisor/GRU_2/while/gru_cell_83/split_1/split_dim
*Supervisor/GRU_2/while/gru_cell_83/split_1SplitV5Supervisor/GRU_2/while/gru_cell_83/BiasAdd_1:output:03Supervisor/GRU_2/while/gru_cell_83/Const_1:output:0=Supervisor/GRU_2/while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*Supervisor/GRU_2/while/gru_cell_83/split_1ѓ
&Supervisor/GRU_2/while/gru_cell_83/addAddV21Supervisor/GRU_2/while/gru_cell_83/split:output:03Supervisor/GRU_2/while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_83/addС
*Supervisor/GRU_2/while/gru_cell_83/SigmoidSigmoid*Supervisor/GRU_2/while/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*Supervisor/GRU_2/while/gru_cell_83/Sigmoidї
(Supervisor/GRU_2/while/gru_cell_83/add_1AddV21Supervisor/GRU_2/while/gru_cell_83/split:output:13Supervisor/GRU_2/while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_83/add_1Ч
,Supervisor/GRU_2/while/gru_cell_83/Sigmoid_1Sigmoid,Supervisor/GRU_2/while/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,Supervisor/GRU_2/while/gru_cell_83/Sigmoid_1№
&Supervisor/GRU_2/while/gru_cell_83/mulMul0Supervisor/GRU_2/while/gru_cell_83/Sigmoid_1:y:03Supervisor/GRU_2/while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_83/mulю
(Supervisor/GRU_2/while/gru_cell_83/add_2AddV21Supervisor/GRU_2/while/gru_cell_83/split:output:2*Supervisor/GRU_2/while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_83/add_2К
'Supervisor/GRU_2/while/gru_cell_83/TanhTanh,Supervisor/GRU_2/while/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'Supervisor/GRU_2/while/gru_cell_83/Tanhу
(Supervisor/GRU_2/while/gru_cell_83/mul_1Mul.Supervisor/GRU_2/while/gru_cell_83/Sigmoid:y:0$supervisor_gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_83/mul_1
(Supervisor/GRU_2/while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(Supervisor/GRU_2/while/gru_cell_83/sub/xь
&Supervisor/GRU_2/while/gru_cell_83/subSub1Supervisor/GRU_2/while/gru_cell_83/sub/x:output:0.Supervisor/GRU_2/while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_83/subц
(Supervisor/GRU_2/while/gru_cell_83/mul_2Mul*Supervisor/GRU_2/while/gru_cell_83/sub:z:0+Supervisor/GRU_2/while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_83/mul_2ы
(Supervisor/GRU_2/while/gru_cell_83/add_3AddV2,Supervisor/GRU_2/while/gru_cell_83/mul_1:z:0,Supervisor/GRU_2/while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_83/add_3Д
;Supervisor/GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$supervisor_gru_2_while_placeholder_1"supervisor_gru_2_while_placeholder,Supervisor/GRU_2/while/gru_cell_83/add_3:z:0*
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
!Supervisor/GRU_2/while/Identity_4Identity,Supervisor/GRU_2/while/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_2/while/Identity_4"
Csupervisor_gru_2_while_gru_cell_83_matmul_1_readvariableop_resourceEsupervisor_gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0"
Asupervisor_gru_2_while_gru_cell_83_matmul_readvariableop_resourceCsupervisor_gru_2_while_gru_cell_83_matmul_readvariableop_resource_0"z
:supervisor_gru_2_while_gru_cell_83_readvariableop_resource<supervisor_gru_2_while_gru_cell_83_readvariableop_resource_0"K
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
Й
и
G__inference_Supervisor_layer_call_and_return_conditional_losses_1686602

inputs
gru_1_1686582
gru_1_1686584
gru_1_1686586
gru_2_1686589
gru_2_1686591
gru_2_1686593
out_1686596
out_1686598
identityЂGRU_1/StatefulPartitionedCallЂGRU_2/StatefulPartitionedCallЂOUT/StatefulPartitionedCall
GRU_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_1686582gru_1_1686584gru_1_1686586*
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
GPU 2J 8 *K
fFRD
B__inference_GRU_1_layer_call_and_return_conditional_losses_16859692
GRU_1/StatefulPartitionedCallН
GRU_2/StatefulPartitionedCallStatefulPartitionedCall&GRU_1/StatefulPartitionedCall:output:0gru_2_1686589gru_2_1686591gru_2_1686593*
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
GPU 2J 8 *K
fFRD
B__inference_GRU_2_layer_call_and_return_conditional_losses_16863162
GRU_2/StatefulPartitionedCallЂ
OUT/StatefulPartitionedCallStatefulPartitionedCall&GRU_2/StatefulPartitionedCall:output:0out_1686596out_1686598*
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
GPU 2J 8 *I
fDRB
@__inference_OUT_layer_call_and_return_conditional_losses_16865362
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
	
Ё
GRU_1_while_cond_1687821(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1687821___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1687821___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1687821___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1687821___redundant_placeholder3
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
О
Ї
 __inference__traced_save_1689799
file_prefix)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop7
3savev2_gru_1_gru_cell_82_kernel_read_readvariableopA
=savev2_gru_1_gru_cell_82_recurrent_kernel_read_readvariableop5
1savev2_gru_1_gru_cell_82_bias_read_readvariableop7
3savev2_gru_2_gru_cell_83_kernel_read_readvariableopA
=savev2_gru_2_gru_cell_83_recurrent_kernel_read_readvariableop5
1savev2_gru_2_gru_cell_83_bias_read_readvariableop
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
value3B1 B+_temp_6e490c33a5f4411588e5f4c72686f6d3/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop3savev2_gru_1_gru_cell_82_kernel_read_readvariableop=savev2_gru_1_gru_cell_82_recurrent_kernel_read_readvariableop1savev2_gru_1_gru_cell_82_bias_read_readvariableop3savev2_gru_2_gru_cell_83_kernel_read_readvariableop=savev2_gru_2_gru_cell_83_recurrent_kernel_read_readvariableop1savev2_gru_2_gru_cell_83_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
	
Ё
GRU_2_while_cond_1687252(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1687252___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1687252___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1687252___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1687252___redundant_placeholder3
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
р	
Џ
-__inference_gru_cell_83_layer_call_fn_1689738

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЄ
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_16853172
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


'__inference_GRU_1_layer_call_fn_1688805

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_1_layer_call_and_return_conditional_losses_16859692
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
оH
и
GRU_1_while_body_1687098(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_82_readvariableop_resource_0<
8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_82_readvariableop_resource:
6gru_1_while_gru_cell_82_matmul_readvariableop_resource<
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_82/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_82/ReadVariableOpВ
GRU_1/while/gru_cell_82/unstackUnpack.GRU_1/while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_82/unstackз
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_82/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_82/MatMulг
GRU_1/while/gru_cell_82/BiasAddBiasAdd(GRU_1/while/gru_cell_82/MatMul:product:0(GRU_1/while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_82/BiasAdd
GRU_1/while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_82/Const
'GRU_1/while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_82/split/split_dim
GRU_1/while/gru_cell_82/splitSplit0GRU_1/while/gru_cell_82/split/split_dim:output:0(GRU_1/while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_82/splitн
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_82/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_82/MatMul_1й
!GRU_1/while/gru_cell_82/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_82/MatMul_1:product:0(GRU_1/while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_82/BiasAdd_1
GRU_1/while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_82/Const_1Ё
)GRU_1/while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_82/split_1/split_dimЫ
GRU_1/while/gru_cell_82/split_1SplitV*GRU_1/while/gru_cell_82/BiasAdd_1:output:0(GRU_1/while/gru_cell_82/Const_1:output:02GRU_1/while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_82/split_1Ч
GRU_1/while/gru_cell_82/addAddV2&GRU_1/while/gru_cell_82/split:output:0(GRU_1/while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add 
GRU_1/while/gru_cell_82/SigmoidSigmoidGRU_1/while/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_82/SigmoidЫ
GRU_1/while/gru_cell_82/add_1AddV2&GRU_1/while/gru_cell_82/split:output:1(GRU_1/while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_1І
!GRU_1/while/gru_cell_82/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_82/Sigmoid_1Ф
GRU_1/while/gru_cell_82/mulMul%GRU_1/while/gru_cell_82/Sigmoid_1:y:0(GRU_1/while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mulТ
GRU_1/while/gru_cell_82/add_2AddV2&GRU_1/while/gru_cell_82/split:output:2GRU_1/while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_2
GRU_1/while/gru_cell_82/TanhTanh!GRU_1/while/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/TanhЗ
GRU_1/while/gru_cell_82/mul_1Mul#GRU_1/while/gru_cell_82/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_1
GRU_1/while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_82/sub/xР
GRU_1/while/gru_cell_82/subSub&GRU_1/while/gru_cell_82/sub/x:output:0#GRU_1/while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/subК
GRU_1/while/gru_cell_82/mul_2MulGRU_1/while/gru_cell_82/sub:z:0 GRU_1/while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_2П
GRU_1/while/gru_cell_82/add_3AddV2!GRU_1/while/gru_cell_82/mul_1:z:0!GRU_1/while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_82/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resource:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_82_matmul_readvariableop_resource8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_82_readvariableop_resource1gru_1_while_gru_cell_82_readvariableop_resource_0"5
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
я
ь
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1689724

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


'__inference_GRU_2_layer_call_fn_1689145

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_2_layer_call_and_return_conditional_losses_16863162
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
р	
Џ
-__inference_gru_cell_82_layer_call_fn_1689644

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЄ
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_16847952
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
е
Џ
while_cond_1685733
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1685733___redundant_placeholder05
1while_while_cond_1685733___redundant_placeholder15
1while_while_cond_1685733___redundant_placeholder25
1while_while_cond_1685733___redundant_placeholder3
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
р	
Џ
-__inference_gru_cell_83_layer_call_fn_1689752

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЄ
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_16853572
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
е
Џ
while_cond_1685053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1685053___redundant_placeholder05
1while_while_cond_1685053___redundant_placeholder15
1while_while_cond_1685053___redundant_placeholder25
1while_while_cond_1685053___redundant_placeholder3
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
е
Џ
while_cond_1685171
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1685171___redundant_placeholder05
1while_while_cond_1685171___redundant_placeholder15
1while_while_cond_1685171___redundant_placeholder25
1while_while_cond_1685171___redundant_placeholder3
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
е
Џ
while_cond_1685615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1685615___redundant_placeholder05
1while_while_cond_1685615___redundant_placeholder15
1while_while_cond_1685615___redundant_placeholder25
1while_while_cond_1685615___redundant_placeholder3
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
Г

'__inference_GRU_1_layer_call_fn_1688476
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_1_layer_call_and_return_conditional_losses_16852362
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
е
Џ
while_cond_1686225
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1686225___redundant_placeholder05
1while_while_cond_1686225___redundant_placeholder15
1while_while_cond_1686225___redundant_placeholder25
1while_while_cond_1686225___redundant_placeholder3
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
ус

G__inference_Supervisor_layer_call_and_return_conditional_losses_1687029
gru_1_input-
)gru_1_gru_cell_82_readvariableop_resource4
0gru_1_gru_cell_82_matmul_readvariableop_resource6
2gru_1_gru_cell_82_matmul_1_readvariableop_resource-
)gru_2_gru_cell_83_readvariableop_resource4
0gru_2_gru_cell_83_matmul_readvariableop_resource6
2gru_2_gru_cell_83_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_82/ReadVariableOpReadVariableOp)gru_1_gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_82/ReadVariableOp 
GRU_1/gru_cell_82/unstackUnpack(GRU_1/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_82/unstackУ
'GRU_1/gru_cell_82/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_82/MatMul/ReadVariableOpС
GRU_1/gru_cell_82/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMulЛ
GRU_1/gru_cell_82/BiasAddBiasAdd"GRU_1/gru_cell_82/MatMul:product:0"GRU_1/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAddt
GRU_1/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_82/Const
!GRU_1/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_82/split/split_dimє
GRU_1/gru_cell_82/splitSplit*GRU_1/gru_cell_82/split/split_dim:output:0"GRU_1/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/splitЩ
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_82/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMul_1С
GRU_1/gru_cell_82/BiasAdd_1BiasAdd$GRU_1/gru_cell_82/MatMul_1:product:0"GRU_1/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAdd_1
GRU_1/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_82/Const_1
#GRU_1/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_82/split_1/split_dim­
GRU_1/gru_cell_82/split_1SplitV$GRU_1/gru_cell_82/BiasAdd_1:output:0"GRU_1/gru_cell_82/Const_1:output:0,GRU_1/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/split_1Џ
GRU_1/gru_cell_82/addAddV2 GRU_1/gru_cell_82/split:output:0"GRU_1/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add
GRU_1/gru_cell_82/SigmoidSigmoidGRU_1/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/SigmoidГ
GRU_1/gru_cell_82/add_1AddV2 GRU_1/gru_cell_82/split:output:1"GRU_1/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_1
GRU_1/gru_cell_82/Sigmoid_1SigmoidGRU_1/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Sigmoid_1Ќ
GRU_1/gru_cell_82/mulMulGRU_1/gru_cell_82/Sigmoid_1:y:0"GRU_1/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mulЊ
GRU_1/gru_cell_82/add_2AddV2 GRU_1/gru_cell_82/split:output:2GRU_1/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_2
GRU_1/gru_cell_82/TanhTanhGRU_1/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Tanh 
GRU_1/gru_cell_82/mul_1MulGRU_1/gru_cell_82/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_1w
GRU_1/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_82/sub/xЈ
GRU_1/gru_cell_82/subSub GRU_1/gru_cell_82/sub/x:output:0GRU_1/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/subЂ
GRU_1/gru_cell_82/mul_2MulGRU_1/gru_cell_82/sub:z:0GRU_1/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_2Ї
GRU_1/gru_cell_82/add_3AddV2GRU_1/gru_cell_82/mul_1:z:0GRU_1/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_3
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
GRU_1/while/loop_counter
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_82_readvariableop_resource0gru_1_gru_cell_82_matmul_readvariableop_resource2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_1_while_body_1686757*$
condR
GRU_1_while_cond_1686756*8
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
 GRU_2/gru_cell_83/ReadVariableOpReadVariableOp)gru_2_gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_83/ReadVariableOp 
GRU_2/gru_cell_83/unstackUnpack(GRU_2/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_83/unstackУ
'GRU_2/gru_cell_83/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_83/MatMul/ReadVariableOpС
GRU_2/gru_cell_83/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMulЛ
GRU_2/gru_cell_83/BiasAddBiasAdd"GRU_2/gru_cell_83/MatMul:product:0"GRU_2/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAddt
GRU_2/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_83/Const
!GRU_2/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_83/split/split_dimє
GRU_2/gru_cell_83/splitSplit*GRU_2/gru_cell_83/split/split_dim:output:0"GRU_2/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/splitЩ
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_83/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMul_1С
GRU_2/gru_cell_83/BiasAdd_1BiasAdd$GRU_2/gru_cell_83/MatMul_1:product:0"GRU_2/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAdd_1
GRU_2/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_83/Const_1
#GRU_2/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_83/split_1/split_dim­
GRU_2/gru_cell_83/split_1SplitV$GRU_2/gru_cell_83/BiasAdd_1:output:0"GRU_2/gru_cell_83/Const_1:output:0,GRU_2/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/split_1Џ
GRU_2/gru_cell_83/addAddV2 GRU_2/gru_cell_83/split:output:0"GRU_2/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add
GRU_2/gru_cell_83/SigmoidSigmoidGRU_2/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/SigmoidГ
GRU_2/gru_cell_83/add_1AddV2 GRU_2/gru_cell_83/split:output:1"GRU_2/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_1
GRU_2/gru_cell_83/Sigmoid_1SigmoidGRU_2/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Sigmoid_1Ќ
GRU_2/gru_cell_83/mulMulGRU_2/gru_cell_83/Sigmoid_1:y:0"GRU_2/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mulЊ
GRU_2/gru_cell_83/add_2AddV2 GRU_2/gru_cell_83/split:output:2GRU_2/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_2
GRU_2/gru_cell_83/TanhTanhGRU_2/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Tanh 
GRU_2/gru_cell_83/mul_1MulGRU_2/gru_cell_83/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_1w
GRU_2/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_83/sub/xЈ
GRU_2/gru_cell_83/subSub GRU_2/gru_cell_83/sub/x:output:0GRU_2/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/subЂ
GRU_2/gru_cell_83/mul_2MulGRU_2/gru_cell_83/sub:z:0GRU_2/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_2Ї
GRU_2/gru_cell_83/add_3AddV2GRU_2/gru_cell_83/mul_1:z:0GRU_2/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_3
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
GRU_2/while/loop_counter
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_83_readvariableop_resource0gru_2_gru_cell_83_matmul_readvariableop_resource2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_2_while_body_1686912*$
condR
GRU_2_while_cond_1686911*8
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
ч
ъ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1684755

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
е
Џ
while_cond_1689043
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1689043___redundant_placeholder05
1while_while_cond_1689043___redundant_placeholder15
1while_while_cond_1689043___redundant_placeholder25
1while_while_cond_1689043___redundant_placeholder3
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
ыW
є
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689134

inputs'
#gru_cell_83_readvariableop_resource.
*gru_cell_83_matmul_readvariableop_resource0
,gru_cell_83_matmul_1_readvariableop_resource
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
gru_cell_83/ReadVariableOpReadVariableOp#gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_83/ReadVariableOp
gru_cell_83/unstackUnpack"gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_83/unstackБ
!gru_cell_83/MatMul/ReadVariableOpReadVariableOp*gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_83/MatMul/ReadVariableOpЉ
gru_cell_83/MatMulMatMulstrided_slice_2:output:0)gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMulЃ
gru_cell_83/BiasAddBiasAddgru_cell_83/MatMul:product:0gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAddh
gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_83/Const
gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split/split_dimм
gru_cell_83/splitSplit$gru_cell_83/split/split_dim:output:0gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/splitЗ
#gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_83/MatMul_1/ReadVariableOpЅ
gru_cell_83/MatMul_1MatMulzeros:output:0+gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMul_1Љ
gru_cell_83/BiasAdd_1BiasAddgru_cell_83/MatMul_1:product:0gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAdd_1
gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_83/Const_1
gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split_1/split_dim
gru_cell_83/split_1SplitVgru_cell_83/BiasAdd_1:output:0gru_cell_83/Const_1:output:0&gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/split_1
gru_cell_83/addAddV2gru_cell_83/split:output:0gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add|
gru_cell_83/SigmoidSigmoidgru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid
gru_cell_83/add_1AddV2gru_cell_83/split:output:1gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_1
gru_cell_83/Sigmoid_1Sigmoidgru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid_1
gru_cell_83/mulMulgru_cell_83/Sigmoid_1:y:0gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul
gru_cell_83/add_2AddV2gru_cell_83/split:output:2gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_2u
gru_cell_83/TanhTanhgru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Tanh
gru_cell_83/mul_1Mulgru_cell_83/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_1k
gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_83/sub/x
gru_cell_83/subSubgru_cell_83/sub/x:output:0gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/sub
gru_cell_83/mul_2Mulgru_cell_83/sub:z:0gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_2
gru_cell_83/add_3AddV2gru_cell_83/mul_1:z:0gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_83_readvariableop_resource*gru_cell_83_matmul_readvariableop_resource,gru_cell_83_matmul_1_readvariableop_resource*
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
bodyR
while_body_1689044*
condR
while_cond_1689043*8
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
ф
z
%__inference_OUT_layer_call_fn_1689536

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
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
GPU 2J 8 *I
fDRB
@__inference_OUT_layer_call_and_return_conditional_losses_16865362
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
е
Џ
while_cond_1688204
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1688204___redundant_placeholder05
1while_while_cond_1688204___redundant_placeholder15
1while_while_cond_1688204___redundant_placeholder25
1while_while_cond_1688204___redundant_placeholder3
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
ыW
є
B__inference_GRU_2_layer_call_and_return_conditional_losses_1688975

inputs'
#gru_cell_83_readvariableop_resource.
*gru_cell_83_matmul_readvariableop_resource0
,gru_cell_83_matmul_1_readvariableop_resource
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
gru_cell_83/ReadVariableOpReadVariableOp#gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_83/ReadVariableOp
gru_cell_83/unstackUnpack"gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_83/unstackБ
!gru_cell_83/MatMul/ReadVariableOpReadVariableOp*gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_83/MatMul/ReadVariableOpЉ
gru_cell_83/MatMulMatMulstrided_slice_2:output:0)gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMulЃ
gru_cell_83/BiasAddBiasAddgru_cell_83/MatMul:product:0gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAddh
gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_83/Const
gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split/split_dimм
gru_cell_83/splitSplit$gru_cell_83/split/split_dim:output:0gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/splitЗ
#gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_83/MatMul_1/ReadVariableOpЅ
gru_cell_83/MatMul_1MatMulzeros:output:0+gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/MatMul_1Љ
gru_cell_83/BiasAdd_1BiasAddgru_cell_83/MatMul_1:product:0gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_83/BiasAdd_1
gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_83/Const_1
gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_83/split_1/split_dim
gru_cell_83/split_1SplitVgru_cell_83/BiasAdd_1:output:0gru_cell_83/Const_1:output:0&gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_83/split_1
gru_cell_83/addAddV2gru_cell_83/split:output:0gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add|
gru_cell_83/SigmoidSigmoidgru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid
gru_cell_83/add_1AddV2gru_cell_83/split:output:1gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_1
gru_cell_83/Sigmoid_1Sigmoidgru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Sigmoid_1
gru_cell_83/mulMulgru_cell_83/Sigmoid_1:y:0gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul
gru_cell_83/add_2AddV2gru_cell_83/split:output:2gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_2u
gru_cell_83/TanhTanhgru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/Tanh
gru_cell_83/mul_1Mulgru_cell_83/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_1k
gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_83/sub/x
gru_cell_83/subSubgru_cell_83/sub/x:output:0gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/sub
gru_cell_83/mul_2Mulgru_cell_83/sub:z:0gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/mul_2
gru_cell_83/add_3AddV2gru_cell_83/mul_1:z:0gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_83/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_83_readvariableop_resource*gru_cell_83_matmul_readvariableop_resource,gru_cell_83_matmul_1_readvariableop_resource*
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
bodyR
while_body_1688885*
condR
while_cond_1688884*8
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
	
Ё
GRU_2_while_cond_1687635(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1687635___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1687635___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1687635___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1687635___redundant_placeholder3
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
е
Џ
while_cond_1688544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1688544___redundant_placeholder05
1while_while_cond_1688544___redundant_placeholder15
1while_while_cond_1688544___redundant_placeholder25
1while_while_cond_1688544___redundant_placeholder3
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
@
Ж
while_body_1689384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_83_readvariableop_resource_06
2while_gru_cell_83_matmul_readvariableop_resource_08
4while_gru_cell_83_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_83_readvariableop_resource4
0while_gru_cell_83_matmul_readvariableop_resource6
2while_gru_cell_83_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_83/ReadVariableOpReadVariableOp+while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_83/ReadVariableOp 
while/gru_cell_83/unstackUnpack(while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_83/unstackХ
'while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_83/MatMul/ReadVariableOpг
while/gru_cell_83/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMulЛ
while/gru_cell_83/BiasAddBiasAdd"while/gru_cell_83/MatMul:product:0"while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAddt
while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_83/Const
!while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_83/split/split_dimє
while/gru_cell_83/splitSplit*while/gru_cell_83/split/split_dim:output:0"while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/splitЫ
)while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_83/MatMul_1/ReadVariableOpМ
while/gru_cell_83/MatMul_1MatMulwhile_placeholder_21while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMul_1С
while/gru_cell_83/BiasAdd_1BiasAdd$while/gru_cell_83/MatMul_1:product:0"while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAdd_1
while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_83/Const_1
#while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_83/split_1/split_dim­
while/gru_cell_83/split_1SplitV$while/gru_cell_83/BiasAdd_1:output:0"while/gru_cell_83/Const_1:output:0,while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/split_1Џ
while/gru_cell_83/addAddV2 while/gru_cell_83/split:output:0"while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add
while/gru_cell_83/SigmoidSigmoidwhile/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/SigmoidГ
while/gru_cell_83/add_1AddV2 while/gru_cell_83/split:output:1"while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_1
while/gru_cell_83/Sigmoid_1Sigmoidwhile/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Sigmoid_1Ќ
while/gru_cell_83/mulMulwhile/gru_cell_83/Sigmoid_1:y:0"while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mulЊ
while/gru_cell_83/add_2AddV2 while/gru_cell_83/split:output:2while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_2
while/gru_cell_83/TanhTanhwhile/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Tanh
while/gru_cell_83/mul_1Mulwhile/gru_cell_83/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_1w
while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_83/sub/xЈ
while/gru_cell_83/subSub while/gru_cell_83/sub/x:output:0while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/subЂ
while/gru_cell_83/mul_2Mulwhile/gru_cell_83/sub:z:0while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_2Ї
while/gru_cell_83/add_3AddV2while/gru_cell_83/mul_1:z:0while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_83/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_83_matmul_1_readvariableop_resource4while_gru_cell_83_matmul_1_readvariableop_resource_0"f
0while_gru_cell_83_matmul_readvariableop_resource2while_gru_cell_83_matmul_readvariableop_resource_0"X
)while_gru_cell_83_readvariableop_resource+while_gru_cell_83_readvariableop_resource_0")
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
е
Џ
while_cond_1688703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1688703___redundant_placeholder05
1while_while_cond_1688703___redundant_placeholder15
1while_while_cond_1688703___redundant_placeholder25
1while_while_cond_1688703___redundant_placeholder3
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
е
Џ
while_cond_1688363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1688363___redundant_placeholder05
1while_while_cond_1688363___redundant_placeholder15
1while_while_cond_1688363___redundant_placeholder25
1while_while_cond_1688363___redundant_placeholder3
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
Ф
ђ
#Supervisor_GRU_1_while_cond_1684410>
:supervisor_gru_1_while_supervisor_gru_1_while_loop_counterD
@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations&
"supervisor_gru_1_while_placeholder(
$supervisor_gru_1_while_placeholder_1(
$supervisor_gru_1_while_placeholder_2@
<supervisor_gru_1_while_less_supervisor_gru_1_strided_slice_1W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1684410___redundant_placeholder0W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1684410___redundant_placeholder1W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1684410___redundant_placeholder2W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1684410___redundant_placeholder3#
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
	
Ё
GRU_2_while_cond_1687976(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1687976___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1687976___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1687976___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1687976___redundant_placeholder3
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
@
Ж
while_body_1688205
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_82_readvariableop_resource_06
2while_gru_cell_82_matmul_readvariableop_resource_08
4while_gru_cell_82_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_82_readvariableop_resource4
0while_gru_cell_82_matmul_readvariableop_resource6
2while_gru_cell_82_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_82/ReadVariableOpReadVariableOp+while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_82/ReadVariableOp 
while/gru_cell_82/unstackUnpack(while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_82/unstackХ
'while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_82/MatMul/ReadVariableOpг
while/gru_cell_82/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMulЛ
while/gru_cell_82/BiasAddBiasAdd"while/gru_cell_82/MatMul:product:0"while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAddt
while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_82/Const
!while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_82/split/split_dimє
while/gru_cell_82/splitSplit*while/gru_cell_82/split/split_dim:output:0"while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/splitЫ
)while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_82/MatMul_1/ReadVariableOpМ
while/gru_cell_82/MatMul_1MatMulwhile_placeholder_21while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMul_1С
while/gru_cell_82/BiasAdd_1BiasAdd$while/gru_cell_82/MatMul_1:product:0"while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAdd_1
while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_82/Const_1
#while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_82/split_1/split_dim­
while/gru_cell_82/split_1SplitV$while/gru_cell_82/BiasAdd_1:output:0"while/gru_cell_82/Const_1:output:0,while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/split_1Џ
while/gru_cell_82/addAddV2 while/gru_cell_82/split:output:0"while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add
while/gru_cell_82/SigmoidSigmoidwhile/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/SigmoidГ
while/gru_cell_82/add_1AddV2 while/gru_cell_82/split:output:1"while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_1
while/gru_cell_82/Sigmoid_1Sigmoidwhile/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Sigmoid_1Ќ
while/gru_cell_82/mulMulwhile/gru_cell_82/Sigmoid_1:y:0"while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mulЊ
while/gru_cell_82/add_2AddV2 while/gru_cell_82/split:output:2while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_2
while/gru_cell_82/TanhTanhwhile/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Tanh
while/gru_cell_82/mul_1Mulwhile/gru_cell_82/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_1w
while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_82/sub/xЈ
while/gru_cell_82/subSub while/gru_cell_82/sub/x:output:0while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/subЂ
while/gru_cell_82/mul_2Mulwhile/gru_cell_82/sub:z:0while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_2Ї
while/gru_cell_82/add_3AddV2while/gru_cell_82/mul_1:z:0while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_82/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_82_matmul_1_readvariableop_resource4while_gru_cell_82_matmul_1_readvariableop_resource_0"f
0while_gru_cell_82_matmul_readvariableop_resource2while_gru_cell_82_matmul_readvariableop_resource_0"X
)while_gru_cell_82_readvariableop_resource+while_gru_cell_82_readvariableop_resource_0")
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
е
Џ
while_cond_1686037
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1686037___redundant_placeholder05
1while_while_cond_1686037___redundant_placeholder15
1while_while_cond_1686037___redundant_placeholder25
1while_while_cond_1686037___redundant_placeholder3
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


'__inference_GRU_1_layer_call_fn_1688816

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_1_layer_call_and_return_conditional_losses_16861282
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
оH
и
GRU_1_while_body_1687481(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_82_readvariableop_resource_0<
8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_82_readvariableop_resource:
6gru_1_while_gru_cell_82_matmul_readvariableop_resource<
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_82/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_82/ReadVariableOpВ
GRU_1/while/gru_cell_82/unstackUnpack.GRU_1/while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_82/unstackз
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_82/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_82/MatMulг
GRU_1/while/gru_cell_82/BiasAddBiasAdd(GRU_1/while/gru_cell_82/MatMul:product:0(GRU_1/while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_82/BiasAdd
GRU_1/while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_82/Const
'GRU_1/while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_82/split/split_dim
GRU_1/while/gru_cell_82/splitSplit0GRU_1/while/gru_cell_82/split/split_dim:output:0(GRU_1/while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_82/splitн
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_82/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_82/MatMul_1й
!GRU_1/while/gru_cell_82/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_82/MatMul_1:product:0(GRU_1/while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_82/BiasAdd_1
GRU_1/while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_82/Const_1Ё
)GRU_1/while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_82/split_1/split_dimЫ
GRU_1/while/gru_cell_82/split_1SplitV*GRU_1/while/gru_cell_82/BiasAdd_1:output:0(GRU_1/while/gru_cell_82/Const_1:output:02GRU_1/while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_82/split_1Ч
GRU_1/while/gru_cell_82/addAddV2&GRU_1/while/gru_cell_82/split:output:0(GRU_1/while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add 
GRU_1/while/gru_cell_82/SigmoidSigmoidGRU_1/while/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_82/SigmoidЫ
GRU_1/while/gru_cell_82/add_1AddV2&GRU_1/while/gru_cell_82/split:output:1(GRU_1/while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_1І
!GRU_1/while/gru_cell_82/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_82/Sigmoid_1Ф
GRU_1/while/gru_cell_82/mulMul%GRU_1/while/gru_cell_82/Sigmoid_1:y:0(GRU_1/while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mulТ
GRU_1/while/gru_cell_82/add_2AddV2&GRU_1/while/gru_cell_82/split:output:2GRU_1/while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_2
GRU_1/while/gru_cell_82/TanhTanh!GRU_1/while/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/TanhЗ
GRU_1/while/gru_cell_82/mul_1Mul#GRU_1/while/gru_cell_82/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_1
GRU_1/while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_82/sub/xР
GRU_1/while/gru_cell_82/subSub&GRU_1/while/gru_cell_82/sub/x:output:0#GRU_1/while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/subК
GRU_1/while/gru_cell_82/mul_2MulGRU_1/while/gru_cell_82/sub:z:0 GRU_1/while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_2П
GRU_1/while/gru_cell_82/add_3AddV2!GRU_1/while/gru_cell_82/mul_1:z:0!GRU_1/while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_82/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resource:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_82_matmul_readvariableop_resource8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_82_readvariableop_resource1gru_1_while_gru_cell_82_readvariableop_resource_0"5
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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688635

inputs'
#gru_cell_82_readvariableop_resource.
*gru_cell_82_matmul_readvariableop_resource0
,gru_cell_82_matmul_1_readvariableop_resource
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
gru_cell_82/ReadVariableOpReadVariableOp#gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_82/ReadVariableOp
gru_cell_82/unstackUnpack"gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_82/unstackБ
!gru_cell_82/MatMul/ReadVariableOpReadVariableOp*gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_82/MatMul/ReadVariableOpЉ
gru_cell_82/MatMulMatMulstrided_slice_2:output:0)gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMulЃ
gru_cell_82/BiasAddBiasAddgru_cell_82/MatMul:product:0gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAddh
gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_82/Const
gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split/split_dimм
gru_cell_82/splitSplit$gru_cell_82/split/split_dim:output:0gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/splitЗ
#gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_82/MatMul_1/ReadVariableOpЅ
gru_cell_82/MatMul_1MatMulzeros:output:0+gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/MatMul_1Љ
gru_cell_82/BiasAdd_1BiasAddgru_cell_82/MatMul_1:product:0gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_82/BiasAdd_1
gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_82/Const_1
gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_82/split_1/split_dim
gru_cell_82/split_1SplitVgru_cell_82/BiasAdd_1:output:0gru_cell_82/Const_1:output:0&gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_82/split_1
gru_cell_82/addAddV2gru_cell_82/split:output:0gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add|
gru_cell_82/SigmoidSigmoidgru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid
gru_cell_82/add_1AddV2gru_cell_82/split:output:1gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_1
gru_cell_82/Sigmoid_1Sigmoidgru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Sigmoid_1
gru_cell_82/mulMulgru_cell_82/Sigmoid_1:y:0gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul
gru_cell_82/add_2AddV2gru_cell_82/split:output:2gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_2u
gru_cell_82/TanhTanhgru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/Tanh
gru_cell_82/mul_1Mulgru_cell_82/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_1k
gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_82/sub/x
gru_cell_82/subSubgru_cell_82/sub/x:output:0gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/sub
gru_cell_82/mul_2Mulgru_cell_82/sub:z:0gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/mul_2
gru_cell_82/add_3AddV2gru_cell_82/mul_1:z:0gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_82/add_3
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
while/loop_counterЌ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_82_readvariableop_resource*gru_cell_82_matmul_readvariableop_resource,gru_cell_82_matmul_1_readvariableop_resource*
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
bodyR
while_body_1688545*
condR
while_cond_1688544*8
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
Ц
о
"__inference__wrapped_model_1684683
gru_1_input8
4supervisor_gru_1_gru_cell_82_readvariableop_resource?
;supervisor_gru_1_gru_cell_82_matmul_readvariableop_resourceA
=supervisor_gru_1_gru_cell_82_matmul_1_readvariableop_resource8
4supervisor_gru_2_gru_cell_83_readvariableop_resource?
;supervisor_gru_2_gru_cell_83_matmul_readvariableop_resourceA
=supervisor_gru_2_gru_cell_83_matmul_1_readvariableop_resource4
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
+Supervisor/GRU_1/gru_cell_82/ReadVariableOpReadVariableOp4supervisor_gru_1_gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02-
+Supervisor/GRU_1/gru_cell_82/ReadVariableOpС
$Supervisor/GRU_1/gru_cell_82/unstackUnpack3Supervisor/GRU_1/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2&
$Supervisor/GRU_1/gru_cell_82/unstackф
2Supervisor/GRU_1/gru_cell_82/MatMul/ReadVariableOpReadVariableOp;supervisor_gru_1_gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype024
2Supervisor/GRU_1/gru_cell_82/MatMul/ReadVariableOpэ
#Supervisor/GRU_1/gru_cell_82/MatMulMatMul)Supervisor/GRU_1/strided_slice_2:output:0:Supervisor/GRU_1/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2%
#Supervisor/GRU_1/gru_cell_82/MatMulч
$Supervisor/GRU_1/gru_cell_82/BiasAddBiasAdd-Supervisor/GRU_1/gru_cell_82/MatMul:product:0-Supervisor/GRU_1/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2&
$Supervisor/GRU_1/gru_cell_82/BiasAdd
"Supervisor/GRU_1/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"Supervisor/GRU_1/gru_cell_82/ConstЇ
,Supervisor/GRU_1/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_1/gru_cell_82/split/split_dim 
"Supervisor/GRU_1/gru_cell_82/splitSplit5Supervisor/GRU_1/gru_cell_82/split/split_dim:output:0-Supervisor/GRU_1/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2$
"Supervisor/GRU_1/gru_cell_82/splitъ
4Supervisor/GRU_1/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp=supervisor_gru_1_gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype026
4Supervisor/GRU_1/gru_cell_82/MatMul_1/ReadVariableOpщ
%Supervisor/GRU_1/gru_cell_82/MatMul_1MatMulSupervisor/GRU_1/zeros:output:0<Supervisor/GRU_1/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2'
%Supervisor/GRU_1/gru_cell_82/MatMul_1э
&Supervisor/GRU_1/gru_cell_82/BiasAdd_1BiasAdd/Supervisor/GRU_1/gru_cell_82/MatMul_1:product:0-Supervisor/GRU_1/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2(
&Supervisor/GRU_1/gru_cell_82/BiasAdd_1Ё
$Supervisor/GRU_1/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2&
$Supervisor/GRU_1/gru_cell_82/Const_1Ћ
.Supervisor/GRU_1/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.Supervisor/GRU_1/gru_cell_82/split_1/split_dimф
$Supervisor/GRU_1/gru_cell_82/split_1SplitV/Supervisor/GRU_1/gru_cell_82/BiasAdd_1:output:0-Supervisor/GRU_1/gru_cell_82/Const_1:output:07Supervisor/GRU_1/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2&
$Supervisor/GRU_1/gru_cell_82/split_1л
 Supervisor/GRU_1/gru_cell_82/addAddV2+Supervisor/GRU_1/gru_cell_82/split:output:0-Supervisor/GRU_1/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_82/addЏ
$Supervisor/GRU_1/gru_cell_82/SigmoidSigmoid$Supervisor/GRU_1/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$Supervisor/GRU_1/gru_cell_82/Sigmoidп
"Supervisor/GRU_1/gru_cell_82/add_1AddV2+Supervisor/GRU_1/gru_cell_82/split:output:1-Supervisor/GRU_1/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_82/add_1Е
&Supervisor/GRU_1/gru_cell_82/Sigmoid_1Sigmoid&Supervisor/GRU_1/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/gru_cell_82/Sigmoid_1и
 Supervisor/GRU_1/gru_cell_82/mulMul*Supervisor/GRU_1/gru_cell_82/Sigmoid_1:y:0-Supervisor/GRU_1/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_82/mulж
"Supervisor/GRU_1/gru_cell_82/add_2AddV2+Supervisor/GRU_1/gru_cell_82/split:output:2$Supervisor/GRU_1/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_82/add_2Ј
!Supervisor/GRU_1/gru_cell_82/TanhTanh&Supervisor/GRU_1/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_1/gru_cell_82/TanhЬ
"Supervisor/GRU_1/gru_cell_82/mul_1Mul(Supervisor/GRU_1/gru_cell_82/Sigmoid:y:0Supervisor/GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_82/mul_1
"Supervisor/GRU_1/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"Supervisor/GRU_1/gru_cell_82/sub/xд
 Supervisor/GRU_1/gru_cell_82/subSub+Supervisor/GRU_1/gru_cell_82/sub/x:output:0(Supervisor/GRU_1/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_82/subЮ
"Supervisor/GRU_1/gru_cell_82/mul_2Mul$Supervisor/GRU_1/gru_cell_82/sub:z:0%Supervisor/GRU_1/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_82/mul_2г
"Supervisor/GRU_1/gru_cell_82/add_3AddV2&Supervisor/GRU_1/gru_cell_82/mul_1:z:0&Supervisor/GRU_1/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_82/add_3Б
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
#Supervisor/GRU_1/while/loop_counter
Supervisor/GRU_1/whileWhile,Supervisor/GRU_1/while/loop_counter:output:02Supervisor/GRU_1/while/maximum_iterations:output:0Supervisor/GRU_1/time:output:0)Supervisor/GRU_1/TensorArrayV2_1:handle:0Supervisor/GRU_1/zeros:output:0)Supervisor/GRU_1/strided_slice_1:output:0HSupervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:04supervisor_gru_1_gru_cell_82_readvariableop_resource;supervisor_gru_1_gru_cell_82_matmul_readvariableop_resource=supervisor_gru_1_gru_cell_82_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#Supervisor_GRU_1_while_body_1684411*/
cond'R%
#Supervisor_GRU_1_while_cond_1684410*8
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
+Supervisor/GRU_2/gru_cell_83/ReadVariableOpReadVariableOp4supervisor_gru_2_gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02-
+Supervisor/GRU_2/gru_cell_83/ReadVariableOpС
$Supervisor/GRU_2/gru_cell_83/unstackUnpack3Supervisor/GRU_2/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2&
$Supervisor/GRU_2/gru_cell_83/unstackф
2Supervisor/GRU_2/gru_cell_83/MatMul/ReadVariableOpReadVariableOp;supervisor_gru_2_gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype024
2Supervisor/GRU_2/gru_cell_83/MatMul/ReadVariableOpэ
#Supervisor/GRU_2/gru_cell_83/MatMulMatMul)Supervisor/GRU_2/strided_slice_2:output:0:Supervisor/GRU_2/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2%
#Supervisor/GRU_2/gru_cell_83/MatMulч
$Supervisor/GRU_2/gru_cell_83/BiasAddBiasAdd-Supervisor/GRU_2/gru_cell_83/MatMul:product:0-Supervisor/GRU_2/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2&
$Supervisor/GRU_2/gru_cell_83/BiasAdd
"Supervisor/GRU_2/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"Supervisor/GRU_2/gru_cell_83/ConstЇ
,Supervisor/GRU_2/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_2/gru_cell_83/split/split_dim 
"Supervisor/GRU_2/gru_cell_83/splitSplit5Supervisor/GRU_2/gru_cell_83/split/split_dim:output:0-Supervisor/GRU_2/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2$
"Supervisor/GRU_2/gru_cell_83/splitъ
4Supervisor/GRU_2/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp=supervisor_gru_2_gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype026
4Supervisor/GRU_2/gru_cell_83/MatMul_1/ReadVariableOpщ
%Supervisor/GRU_2/gru_cell_83/MatMul_1MatMulSupervisor/GRU_2/zeros:output:0<Supervisor/GRU_2/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2'
%Supervisor/GRU_2/gru_cell_83/MatMul_1э
&Supervisor/GRU_2/gru_cell_83/BiasAdd_1BiasAdd/Supervisor/GRU_2/gru_cell_83/MatMul_1:product:0-Supervisor/GRU_2/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2(
&Supervisor/GRU_2/gru_cell_83/BiasAdd_1Ё
$Supervisor/GRU_2/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2&
$Supervisor/GRU_2/gru_cell_83/Const_1Ћ
.Supervisor/GRU_2/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.Supervisor/GRU_2/gru_cell_83/split_1/split_dimф
$Supervisor/GRU_2/gru_cell_83/split_1SplitV/Supervisor/GRU_2/gru_cell_83/BiasAdd_1:output:0-Supervisor/GRU_2/gru_cell_83/Const_1:output:07Supervisor/GRU_2/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2&
$Supervisor/GRU_2/gru_cell_83/split_1л
 Supervisor/GRU_2/gru_cell_83/addAddV2+Supervisor/GRU_2/gru_cell_83/split:output:0-Supervisor/GRU_2/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_83/addЏ
$Supervisor/GRU_2/gru_cell_83/SigmoidSigmoid$Supervisor/GRU_2/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$Supervisor/GRU_2/gru_cell_83/Sigmoidп
"Supervisor/GRU_2/gru_cell_83/add_1AddV2+Supervisor/GRU_2/gru_cell_83/split:output:1-Supervisor/GRU_2/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_83/add_1Е
&Supervisor/GRU_2/gru_cell_83/Sigmoid_1Sigmoid&Supervisor/GRU_2/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/gru_cell_83/Sigmoid_1и
 Supervisor/GRU_2/gru_cell_83/mulMul*Supervisor/GRU_2/gru_cell_83/Sigmoid_1:y:0-Supervisor/GRU_2/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_83/mulж
"Supervisor/GRU_2/gru_cell_83/add_2AddV2+Supervisor/GRU_2/gru_cell_83/split:output:2$Supervisor/GRU_2/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_83/add_2Ј
!Supervisor/GRU_2/gru_cell_83/TanhTanh&Supervisor/GRU_2/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_2/gru_cell_83/TanhЬ
"Supervisor/GRU_2/gru_cell_83/mul_1Mul(Supervisor/GRU_2/gru_cell_83/Sigmoid:y:0Supervisor/GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_83/mul_1
"Supervisor/GRU_2/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"Supervisor/GRU_2/gru_cell_83/sub/xд
 Supervisor/GRU_2/gru_cell_83/subSub+Supervisor/GRU_2/gru_cell_83/sub/x:output:0(Supervisor/GRU_2/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_83/subЮ
"Supervisor/GRU_2/gru_cell_83/mul_2Mul$Supervisor/GRU_2/gru_cell_83/sub:z:0%Supervisor/GRU_2/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_83/mul_2г
"Supervisor/GRU_2/gru_cell_83/add_3AddV2&Supervisor/GRU_2/gru_cell_83/mul_1:z:0&Supervisor/GRU_2/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_83/add_3Б
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
#Supervisor/GRU_2/while/loop_counter
Supervisor/GRU_2/whileWhile,Supervisor/GRU_2/while/loop_counter:output:02Supervisor/GRU_2/while/maximum_iterations:output:0Supervisor/GRU_2/time:output:0)Supervisor/GRU_2/TensorArrayV2_1:handle:0Supervisor/GRU_2/zeros:output:0)Supervisor/GRU_2/strided_slice_1:output:0HSupervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:04supervisor_gru_2_gru_cell_83_readvariableop_resource;supervisor_gru_2_gru_cell_83_matmul_readvariableop_resource=supervisor_gru_2_gru_cell_83_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#Supervisor_GRU_2_while_body_1684566*/
cond'R%
#Supervisor_GRU_2_while_cond_1684565*8
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
&
ч
#__inference__traced_restore_1689833
file_prefix
assignvariableop_out_kernel
assignvariableop_1_out_bias/
+assignvariableop_2_gru_1_gru_cell_82_kernel9
5assignvariableop_3_gru_1_gru_cell_82_recurrent_kernel-
)assignvariableop_4_gru_1_gru_cell_82_bias/
+assignvariableop_5_gru_2_gru_cell_83_kernel9
5assignvariableop_6_gru_2_gru_cell_83_recurrent_kernel-
)assignvariableop_7_gru_2_gru_cell_83_bias

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
AssignVariableOp_2AssignVariableOp+assignvariableop_2_gru_1_gru_cell_82_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3К
AssignVariableOp_3AssignVariableOp5assignvariableop_3_gru_1_gru_cell_82_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ў
AssignVariableOp_4AssignVariableOp)assignvariableop_4_gru_1_gru_cell_82_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5А
AssignVariableOp_5AssignVariableOp+assignvariableop_5_gru_2_gru_cell_83_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6К
AssignVariableOp_6AssignVariableOp5assignvariableop_6_gru_2_gru_cell_83_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ў
AssignVariableOp_7AssignVariableOp)assignvariableop_7_gru_2_gru_cell_83_biasIdentity_7:output:0"/device:CPU:0*
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
@
Ж
while_body_1688364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_82_readvariableop_resource_06
2while_gru_cell_82_matmul_readvariableop_resource_08
4while_gru_cell_82_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_82_readvariableop_resource4
0while_gru_cell_82_matmul_readvariableop_resource6
2while_gru_cell_82_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_82/ReadVariableOpReadVariableOp+while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_82/ReadVariableOp 
while/gru_cell_82/unstackUnpack(while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_82/unstackХ
'while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_82/MatMul/ReadVariableOpг
while/gru_cell_82/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMulЛ
while/gru_cell_82/BiasAddBiasAdd"while/gru_cell_82/MatMul:product:0"while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAddt
while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_82/Const
!while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_82/split/split_dimє
while/gru_cell_82/splitSplit*while/gru_cell_82/split/split_dim:output:0"while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/splitЫ
)while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_82/MatMul_1/ReadVariableOpМ
while/gru_cell_82/MatMul_1MatMulwhile_placeholder_21while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMul_1С
while/gru_cell_82/BiasAdd_1BiasAdd$while/gru_cell_82/MatMul_1:product:0"while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAdd_1
while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_82/Const_1
#while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_82/split_1/split_dim­
while/gru_cell_82/split_1SplitV$while/gru_cell_82/BiasAdd_1:output:0"while/gru_cell_82/Const_1:output:0,while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/split_1Џ
while/gru_cell_82/addAddV2 while/gru_cell_82/split:output:0"while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add
while/gru_cell_82/SigmoidSigmoidwhile/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/SigmoidГ
while/gru_cell_82/add_1AddV2 while/gru_cell_82/split:output:1"while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_1
while/gru_cell_82/Sigmoid_1Sigmoidwhile/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Sigmoid_1Ќ
while/gru_cell_82/mulMulwhile/gru_cell_82/Sigmoid_1:y:0"while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mulЊ
while/gru_cell_82/add_2AddV2 while/gru_cell_82/split:output:2while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_2
while/gru_cell_82/TanhTanhwhile/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Tanh
while/gru_cell_82/mul_1Mulwhile/gru_cell_82/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_1w
while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_82/sub/xЈ
while/gru_cell_82/subSub while/gru_cell_82/sub/x:output:0while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/subЂ
while/gru_cell_82/mul_2Mulwhile/gru_cell_82/sub:z:0while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_2Ї
while/gru_cell_82/add_3AddV2while/gru_cell_82/mul_1:z:0while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_82/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_82_matmul_1_readvariableop_resource4while_gru_cell_82_matmul_1_readvariableop_resource_0"f
0while_gru_cell_82_matmul_readvariableop_resource2while_gru_cell_82_matmul_readvariableop_resource_0"X
)while_gru_cell_82_readvariableop_resource+while_gru_cell_82_readvariableop_resource_0")
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
оH
и
GRU_2_while_body_1686912(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_83_readvariableop_resource_0<
8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_83_readvariableop_resource:
6gru_2_while_gru_cell_83_matmul_readvariableop_resource<
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_83/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_83/ReadVariableOpВ
GRU_2/while/gru_cell_83/unstackUnpack.GRU_2/while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_83/unstackз
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_83/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_83/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_83/MatMulг
GRU_2/while/gru_cell_83/BiasAddBiasAdd(GRU_2/while/gru_cell_83/MatMul:product:0(GRU_2/while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_83/BiasAdd
GRU_2/while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_83/Const
'GRU_2/while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_83/split/split_dim
GRU_2/while/gru_cell_83/splitSplit0GRU_2/while/gru_cell_83/split/split_dim:output:0(GRU_2/while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_83/splitн
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_83/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_83/MatMul_1й
!GRU_2/while/gru_cell_83/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_83/MatMul_1:product:0(GRU_2/while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_83/BiasAdd_1
GRU_2/while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_83/Const_1Ё
)GRU_2/while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_83/split_1/split_dimЫ
GRU_2/while/gru_cell_83/split_1SplitV*GRU_2/while/gru_cell_83/BiasAdd_1:output:0(GRU_2/while/gru_cell_83/Const_1:output:02GRU_2/while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_83/split_1Ч
GRU_2/while/gru_cell_83/addAddV2&GRU_2/while/gru_cell_83/split:output:0(GRU_2/while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add 
GRU_2/while/gru_cell_83/SigmoidSigmoidGRU_2/while/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_83/SigmoidЫ
GRU_2/while/gru_cell_83/add_1AddV2&GRU_2/while/gru_cell_83/split:output:1(GRU_2/while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_1І
!GRU_2/while/gru_cell_83/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_83/Sigmoid_1Ф
GRU_2/while/gru_cell_83/mulMul%GRU_2/while/gru_cell_83/Sigmoid_1:y:0(GRU_2/while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mulТ
GRU_2/while/gru_cell_83/add_2AddV2&GRU_2/while/gru_cell_83/split:output:2GRU_2/while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_2
GRU_2/while/gru_cell_83/TanhTanh!GRU_2/while/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/TanhЗ
GRU_2/while/gru_cell_83/mul_1Mul#GRU_2/while/gru_cell_83/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_1
GRU_2/while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_83/sub/xР
GRU_2/while/gru_cell_83/subSub&GRU_2/while/gru_cell_83/sub/x:output:0#GRU_2/while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/subК
GRU_2/while/gru_cell_83/mul_2MulGRU_2/while/gru_cell_83/sub:z:0 GRU_2/while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/mul_2П
GRU_2/while/gru_cell_83/add_3AddV2!GRU_2/while/gru_cell_83/mul_1:z:0!GRU_2/while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_83/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_83/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_83_matmul_1_readvariableop_resource:gru_2_while_gru_cell_83_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_83_matmul_readvariableop_resource8gru_2_while_gru_cell_83_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_83_readvariableop_resource1gru_2_while_gru_cell_83_readvariableop_resource_0"5
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
я
ь
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1689616

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
Ћ
@__inference_OUT_layer_call_and_return_conditional_losses_1689527

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
я
ь
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1689576

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
і<
к
B__inference_GRU_1_layer_call_and_return_conditional_losses_1685236

inputs
gru_cell_82_1685160
gru_cell_82_1685162
gru_cell_82_1685164
identityЂ#gru_cell_82/StatefulPartitionedCallЂwhileD
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
strided_slice_2є
#gru_cell_82/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_82_1685160gru_cell_82_1685162gru_cell_82_1685164*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_16847952%
#gru_cell_82/StatefulPartitionedCall
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
while/loop_counterь
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_82_1685160gru_cell_82_1685162gru_cell_82_1685164*
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
bodyR
while_body_1685172*
condR
while_cond_1685171*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_82/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_82/StatefulPartitionedCall#gru_cell_82/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е
Џ
while_cond_1685878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1685878___redundant_placeholder05
1while_while_cond_1685878___redundant_placeholder15
1while_while_cond_1685878___redundant_placeholder25
1while_while_cond_1685878___redundant_placeholder3
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
оH
и
GRU_1_while_body_1687822(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_82_readvariableop_resource_0<
8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_82_readvariableop_resource:
6gru_1_while_gru_cell_82_matmul_readvariableop_resource<
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_82/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_82/ReadVariableOpВ
GRU_1/while/gru_cell_82/unstackUnpack.GRU_1/while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_82/unstackз
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_82/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_82/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_82/MatMulг
GRU_1/while/gru_cell_82/BiasAddBiasAdd(GRU_1/while/gru_cell_82/MatMul:product:0(GRU_1/while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_82/BiasAdd
GRU_1/while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_82/Const
'GRU_1/while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_82/split/split_dim
GRU_1/while/gru_cell_82/splitSplit0GRU_1/while/gru_cell_82/split/split_dim:output:0(GRU_1/while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_82/splitн
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_82/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_82/MatMul_1й
!GRU_1/while/gru_cell_82/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_82/MatMul_1:product:0(GRU_1/while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_82/BiasAdd_1
GRU_1/while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_82/Const_1Ё
)GRU_1/while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_82/split_1/split_dimЫ
GRU_1/while/gru_cell_82/split_1SplitV*GRU_1/while/gru_cell_82/BiasAdd_1:output:0(GRU_1/while/gru_cell_82/Const_1:output:02GRU_1/while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_82/split_1Ч
GRU_1/while/gru_cell_82/addAddV2&GRU_1/while/gru_cell_82/split:output:0(GRU_1/while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add 
GRU_1/while/gru_cell_82/SigmoidSigmoidGRU_1/while/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_82/SigmoidЫ
GRU_1/while/gru_cell_82/add_1AddV2&GRU_1/while/gru_cell_82/split:output:1(GRU_1/while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_1І
!GRU_1/while/gru_cell_82/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_82/Sigmoid_1Ф
GRU_1/while/gru_cell_82/mulMul%GRU_1/while/gru_cell_82/Sigmoid_1:y:0(GRU_1/while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mulТ
GRU_1/while/gru_cell_82/add_2AddV2&GRU_1/while/gru_cell_82/split:output:2GRU_1/while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_2
GRU_1/while/gru_cell_82/TanhTanh!GRU_1/while/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/TanhЗ
GRU_1/while/gru_cell_82/mul_1Mul#GRU_1/while/gru_cell_82/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_1
GRU_1/while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_82/sub/xР
GRU_1/while/gru_cell_82/subSub&GRU_1/while/gru_cell_82/sub/x:output:0#GRU_1/while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/subК
GRU_1/while/gru_cell_82/mul_2MulGRU_1/while/gru_cell_82/sub:z:0 GRU_1/while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/mul_2П
GRU_1/while/gru_cell_82/add_3AddV2!GRU_1/while/gru_cell_82/mul_1:z:0!GRU_1/while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_82/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_82/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_82_matmul_1_readvariableop_resource:gru_1_while_gru_cell_82_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_82_matmul_readvariableop_resource8gru_1_while_gru_cell_82_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_82_readvariableop_resource1gru_1_while_gru_cell_82_readvariableop_resource_0"5
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
і<
к
B__inference_GRU_1_layer_call_and_return_conditional_losses_1685118

inputs
gru_cell_82_1685042
gru_cell_82_1685044
gru_cell_82_1685046
identityЂ#gru_cell_82/StatefulPartitionedCallЂwhileD
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
strided_slice_2є
#gru_cell_82/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_82_1685042gru_cell_82_1685044gru_cell_82_1685046*
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
GPU 2J 8 *Q
fLRJ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_16847552%
#gru_cell_82/StatefulPartitionedCall
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
while/loop_counterь
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_82_1685042gru_cell_82_1685044gru_cell_82_1685046*
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
bodyR
while_body_1685054*
condR
while_cond_1685053*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_82/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_82/StatefulPartitionedCall#gru_cell_82/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г

'__inference_GRU_1_layer_call_fn_1688465
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_1_layer_call_and_return_conditional_losses_16851182
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
Г

'__inference_GRU_2_layer_call_fn_1689496
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_GRU_2_layer_call_and_return_conditional_losses_16857982
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
@
Ж
while_body_1688545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_82_readvariableop_resource_06
2while_gru_cell_82_matmul_readvariableop_resource_08
4while_gru_cell_82_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_82_readvariableop_resource4
0while_gru_cell_82_matmul_readvariableop_resource6
2while_gru_cell_82_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_82/ReadVariableOpReadVariableOp+while_gru_cell_82_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_82/ReadVariableOp 
while/gru_cell_82/unstackUnpack(while/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_82/unstackХ
'while/gru_cell_82/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_82_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_82/MatMul/ReadVariableOpг
while/gru_cell_82/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMulЛ
while/gru_cell_82/BiasAddBiasAdd"while/gru_cell_82/MatMul:product:0"while/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAddt
while/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_82/Const
!while/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_82/split/split_dimє
while/gru_cell_82/splitSplit*while/gru_cell_82/split/split_dim:output:0"while/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/splitЫ
)while/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_82_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_82/MatMul_1/ReadVariableOpМ
while/gru_cell_82/MatMul_1MatMulwhile_placeholder_21while/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/MatMul_1С
while/gru_cell_82/BiasAdd_1BiasAdd$while/gru_cell_82/MatMul_1:product:0"while/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_82/BiasAdd_1
while/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_82/Const_1
#while/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_82/split_1/split_dim­
while/gru_cell_82/split_1SplitV$while/gru_cell_82/BiasAdd_1:output:0"while/gru_cell_82/Const_1:output:0,while/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_82/split_1Џ
while/gru_cell_82/addAddV2 while/gru_cell_82/split:output:0"while/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add
while/gru_cell_82/SigmoidSigmoidwhile/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/SigmoidГ
while/gru_cell_82/add_1AddV2 while/gru_cell_82/split:output:1"while/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_1
while/gru_cell_82/Sigmoid_1Sigmoidwhile/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Sigmoid_1Ќ
while/gru_cell_82/mulMulwhile/gru_cell_82/Sigmoid_1:y:0"while/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mulЊ
while/gru_cell_82/add_2AddV2 while/gru_cell_82/split:output:2while/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_2
while/gru_cell_82/TanhTanhwhile/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/Tanh
while/gru_cell_82/mul_1Mulwhile/gru_cell_82/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_1w
while/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_82/sub/xЈ
while/gru_cell_82/subSub while/gru_cell_82/sub/x:output:0while/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/subЂ
while/gru_cell_82/mul_2Mulwhile/gru_cell_82/sub:z:0while/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/mul_2Ї
while/gru_cell_82/add_3AddV2while/gru_cell_82/mul_1:z:0while/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_82/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_82/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_82/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_82_matmul_1_readvariableop_resource4while_gru_cell_82_matmul_1_readvariableop_resource_0"f
0while_gru_cell_82_matmul_readvariableop_resource2while_gru_cell_82_matmul_readvariableop_resource_0"X
)while_gru_cell_82_readvariableop_resource+while_gru_cell_82_readvariableop_resource_0")
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
Яс

G__inference_Supervisor_layer_call_and_return_conditional_losses_1687753

inputs-
)gru_1_gru_cell_82_readvariableop_resource4
0gru_1_gru_cell_82_matmul_readvariableop_resource6
2gru_1_gru_cell_82_matmul_1_readvariableop_resource-
)gru_2_gru_cell_83_readvariableop_resource4
0gru_2_gru_cell_83_matmul_readvariableop_resource6
2gru_2_gru_cell_83_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_82/ReadVariableOpReadVariableOp)gru_1_gru_cell_82_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_82/ReadVariableOp 
GRU_1/gru_cell_82/unstackUnpack(GRU_1/gru_cell_82/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_82/unstackУ
'GRU_1/gru_cell_82/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_82_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_82/MatMul/ReadVariableOpС
GRU_1/gru_cell_82/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMulЛ
GRU_1/gru_cell_82/BiasAddBiasAdd"GRU_1/gru_cell_82/MatMul:product:0"GRU_1/gru_cell_82/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAddt
GRU_1/gru_cell_82/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_82/Const
!GRU_1/gru_cell_82/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_82/split/split_dimє
GRU_1/gru_cell_82/splitSplit*GRU_1/gru_cell_82/split/split_dim:output:0"GRU_1/gru_cell_82/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/splitЩ
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_82/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_82/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_82/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/MatMul_1С
GRU_1/gru_cell_82/BiasAdd_1BiasAdd$GRU_1/gru_cell_82/MatMul_1:product:0"GRU_1/gru_cell_82/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_82/BiasAdd_1
GRU_1/gru_cell_82/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_82/Const_1
#GRU_1/gru_cell_82/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_82/split_1/split_dim­
GRU_1/gru_cell_82/split_1SplitV$GRU_1/gru_cell_82/BiasAdd_1:output:0"GRU_1/gru_cell_82/Const_1:output:0,GRU_1/gru_cell_82/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_82/split_1Џ
GRU_1/gru_cell_82/addAddV2 GRU_1/gru_cell_82/split:output:0"GRU_1/gru_cell_82/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add
GRU_1/gru_cell_82/SigmoidSigmoidGRU_1/gru_cell_82/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/SigmoidГ
GRU_1/gru_cell_82/add_1AddV2 GRU_1/gru_cell_82/split:output:1"GRU_1/gru_cell_82/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_1
GRU_1/gru_cell_82/Sigmoid_1SigmoidGRU_1/gru_cell_82/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Sigmoid_1Ќ
GRU_1/gru_cell_82/mulMulGRU_1/gru_cell_82/Sigmoid_1:y:0"GRU_1/gru_cell_82/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mulЊ
GRU_1/gru_cell_82/add_2AddV2 GRU_1/gru_cell_82/split:output:2GRU_1/gru_cell_82/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_2
GRU_1/gru_cell_82/TanhTanhGRU_1/gru_cell_82/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/Tanh 
GRU_1/gru_cell_82/mul_1MulGRU_1/gru_cell_82/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_1w
GRU_1/gru_cell_82/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_82/sub/xЈ
GRU_1/gru_cell_82/subSub GRU_1/gru_cell_82/sub/x:output:0GRU_1/gru_cell_82/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/subЂ
GRU_1/gru_cell_82/mul_2MulGRU_1/gru_cell_82/sub:z:0GRU_1/gru_cell_82/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/mul_2Ї
GRU_1/gru_cell_82/add_3AddV2GRU_1/gru_cell_82/mul_1:z:0GRU_1/gru_cell_82/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_82/add_3
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
GRU_1/while/loop_counter
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_82_readvariableop_resource0gru_1_gru_cell_82_matmul_readvariableop_resource2gru_1_gru_cell_82_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_1_while_body_1687481*$
condR
GRU_1_while_cond_1687480*8
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
 GRU_2/gru_cell_83/ReadVariableOpReadVariableOp)gru_2_gru_cell_83_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_83/ReadVariableOp 
GRU_2/gru_cell_83/unstackUnpack(GRU_2/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_83/unstackУ
'GRU_2/gru_cell_83/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_83_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_83/MatMul/ReadVariableOpС
GRU_2/gru_cell_83/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMulЛ
GRU_2/gru_cell_83/BiasAddBiasAdd"GRU_2/gru_cell_83/MatMul:product:0"GRU_2/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAddt
GRU_2/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_83/Const
!GRU_2/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_83/split/split_dimє
GRU_2/gru_cell_83/splitSplit*GRU_2/gru_cell_83/split/split_dim:output:0"GRU_2/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/splitЩ
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_83/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_83/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/MatMul_1С
GRU_2/gru_cell_83/BiasAdd_1BiasAdd$GRU_2/gru_cell_83/MatMul_1:product:0"GRU_2/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_83/BiasAdd_1
GRU_2/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_83/Const_1
#GRU_2/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_83/split_1/split_dim­
GRU_2/gru_cell_83/split_1SplitV$GRU_2/gru_cell_83/BiasAdd_1:output:0"GRU_2/gru_cell_83/Const_1:output:0,GRU_2/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_83/split_1Џ
GRU_2/gru_cell_83/addAddV2 GRU_2/gru_cell_83/split:output:0"GRU_2/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add
GRU_2/gru_cell_83/SigmoidSigmoidGRU_2/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/SigmoidГ
GRU_2/gru_cell_83/add_1AddV2 GRU_2/gru_cell_83/split:output:1"GRU_2/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_1
GRU_2/gru_cell_83/Sigmoid_1SigmoidGRU_2/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Sigmoid_1Ќ
GRU_2/gru_cell_83/mulMulGRU_2/gru_cell_83/Sigmoid_1:y:0"GRU_2/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mulЊ
GRU_2/gru_cell_83/add_2AddV2 GRU_2/gru_cell_83/split:output:2GRU_2/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_2
GRU_2/gru_cell_83/TanhTanhGRU_2/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/Tanh 
GRU_2/gru_cell_83/mul_1MulGRU_2/gru_cell_83/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_1w
GRU_2/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_83/sub/xЈ
GRU_2/gru_cell_83/subSub GRU_2/gru_cell_83/sub/x:output:0GRU_2/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/subЂ
GRU_2/gru_cell_83/mul_2MulGRU_2/gru_cell_83/sub:z:0GRU_2/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/mul_2Ї
GRU_2/gru_cell_83/add_3AddV2GRU_2/gru_cell_83/mul_1:z:0GRU_2/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_83/add_3
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
GRU_2/while/loop_counter
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_83_readvariableop_resource0gru_2_gru_cell_83_matmul_readvariableop_resource2gru_2_gru_cell_83_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*$
bodyR
GRU_2_while_body_1687636*$
condR
GRU_2_while_cond_1687635*8
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

й
%__inference_signature_wrapper_1686688
gru_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЉ
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
GPU 2J 8 *+
f&R$
"__inference__wrapped_model_16846832
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
@
Ж
while_body_1686226
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_83_readvariableop_resource_06
2while_gru_cell_83_matmul_readvariableop_resource_08
4while_gru_cell_83_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_83_readvariableop_resource4
0while_gru_cell_83_matmul_readvariableop_resource6
2while_gru_cell_83_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_83/ReadVariableOpReadVariableOp+while_gru_cell_83_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_83/ReadVariableOp 
while/gru_cell_83/unstackUnpack(while/gru_cell_83/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_83/unstackХ
'while/gru_cell_83/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_83_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_83/MatMul/ReadVariableOpг
while/gru_cell_83/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMulЛ
while/gru_cell_83/BiasAddBiasAdd"while/gru_cell_83/MatMul:product:0"while/gru_cell_83/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAddt
while/gru_cell_83/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_83/Const
!while/gru_cell_83/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_83/split/split_dimє
while/gru_cell_83/splitSplit*while/gru_cell_83/split/split_dim:output:0"while/gru_cell_83/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/splitЫ
)while/gru_cell_83/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_83_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_83/MatMul_1/ReadVariableOpМ
while/gru_cell_83/MatMul_1MatMulwhile_placeholder_21while/gru_cell_83/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/MatMul_1С
while/gru_cell_83/BiasAdd_1BiasAdd$while/gru_cell_83/MatMul_1:product:0"while/gru_cell_83/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_83/BiasAdd_1
while/gru_cell_83/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_83/Const_1
#while/gru_cell_83/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_83/split_1/split_dim­
while/gru_cell_83/split_1SplitV$while/gru_cell_83/BiasAdd_1:output:0"while/gru_cell_83/Const_1:output:0,while/gru_cell_83/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_83/split_1Џ
while/gru_cell_83/addAddV2 while/gru_cell_83/split:output:0"while/gru_cell_83/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add
while/gru_cell_83/SigmoidSigmoidwhile/gru_cell_83/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/SigmoidГ
while/gru_cell_83/add_1AddV2 while/gru_cell_83/split:output:1"while/gru_cell_83/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_1
while/gru_cell_83/Sigmoid_1Sigmoidwhile/gru_cell_83/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Sigmoid_1Ќ
while/gru_cell_83/mulMulwhile/gru_cell_83/Sigmoid_1:y:0"while/gru_cell_83/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mulЊ
while/gru_cell_83/add_2AddV2 while/gru_cell_83/split:output:2while/gru_cell_83/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_2
while/gru_cell_83/TanhTanhwhile/gru_cell_83/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/Tanh
while/gru_cell_83/mul_1Mulwhile/gru_cell_83/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_1w
while/gru_cell_83/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_83/sub/xЈ
while/gru_cell_83/subSub while/gru_cell_83/sub/x:output:0while/gru_cell_83/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/subЂ
while/gru_cell_83/mul_2Mulwhile/gru_cell_83/sub:z:0while/gru_cell_83/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/mul_2Ї
while/gru_cell_83/add_3AddV2while/gru_cell_83/mul_1:z:0while/gru_cell_83/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_83/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_83/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_83/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_83_matmul_1_readvariableop_resource4while_gru_cell_83_matmul_1_readvariableop_resource_0"f
0while_gru_cell_83_matmul_readvariableop_resource2while_gru_cell_83_matmul_readvariableop_resource_0"X
)while_gru_cell_83_readvariableop_resource+while_gru_cell_83_readvariableop_resource_0")
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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Фи
+
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*N&call_and_return_all_conditional_losses
O__call__
P_default_save_signature"ъ(
_tf_keras_sequentialЫ({"class_name": "Sequential", "name": "Supervisor", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Supervisor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "GRU_1_input"}}, {"class_name": "GRU", "config": {"name": "GRU_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "GRU", "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "OUT", "trainable": true, "dtype": "float32", "units": 20, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Supervisor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "GRU_1_input"}}, {"class_name": "GRU", "config": {"name": "GRU_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "GRU", "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "OUT", "trainable": true, "dtype": "float32", "units": 20, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
м
	cell

_inbound_nodes

state_spec
_outbound_nodes
regularization_losses
	variables
trainable_variables
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"

_tf_keras_rnn_layerь	{"class_name": "GRU", "name": "GRU_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GRU_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}
м
cell
_inbound_nodes

state_spec
_outbound_nodes
regularization_losses
	variables
trainable_variables
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"

_tf_keras_rnn_layerь	{"class_name": "GRU", "name": "GRU_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}

_inbound_nodes

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"Щ
_tf_keras_layerЏ{"class_name": "Dense", "name": "OUT", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "OUT", "trainable": true, "dtype": "float32", "units": 20, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}
 "
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
Ъ
regularization_losses

&layers
	variables
'layer_metrics
(non_trainable_variables
)metrics
*layer_regularization_losses
trainable_variables
O__call__
P_default_save_signature
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
Ђ

 kernel
!recurrent_kernel
"bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_82", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_82", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
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
Й
regularization_losses

/layers

0states
	variables
1layer_metrics
2non_trainable_variables
3metrics
4layer_regularization_losses
trainable_variables
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
Ђ

#kernel
$recurrent_kernel
%bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_83", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_83", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
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
Й
regularization_losses

9layers

:states
	variables
;layer_metrics
<non_trainable_variables
=metrics
>layer_regularization_losses
trainable_variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2
OUT/kernel
:2OUT/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses

?layers
	variables
@layer_metrics
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
trainable_variables
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
*:(<2GRU_1/gru_cell_82/kernel
4:2<2"GRU_1/gru_cell_82/recurrent_kernel
(:&<2GRU_1/gru_cell_82/bias
*:(<2GRU_2/gru_cell_83/kernel
4:2<2"GRU_2/gru_cell_83/recurrent_kernel
(:&<2GRU_2/gru_cell_83/bias
5
0
1
2"
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
­
+regularization_losses

Dlayers
,	variables
Elayer_metrics
Fnon_trainable_variables
Gmetrics
Hlayer_regularization_losses
-trainable_variables
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
'
	0"
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
­
5regularization_losses

Ilayers
6	variables
Jlayer_metrics
Knon_trainable_variables
Lmetrics
Mlayer_regularization_losses
7trainable_variables
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
ъ2ч
G__inference_Supervisor_layer_call_and_return_conditional_losses_1688094
G__inference_Supervisor_layer_call_and_return_conditional_losses_1687370
G__inference_Supervisor_layer_call_and_return_conditional_losses_1687029
G__inference_Supervisor_layer_call_and_return_conditional_losses_1687753Р
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
ў2ћ
,__inference_Supervisor_layer_call_fn_1687391
,__inference_Supervisor_layer_call_fn_1688115
,__inference_Supervisor_layer_call_fn_1688136
,__inference_Supervisor_layer_call_fn_1687412Р
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
ш2х
"__inference__wrapped_model_1684683О
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
ы2ш
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688295
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688635
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688794
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688454е
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
џ2ќ
'__inference_GRU_1_layer_call_fn_1688476
'__inference_GRU_1_layer_call_fn_1688816
'__inference_GRU_1_layer_call_fn_1688805
'__inference_GRU_1_layer_call_fn_1688465е
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
ы2ш
B__inference_GRU_2_layer_call_and_return_conditional_losses_1688975
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689134
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689474
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689315е
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
џ2ќ
'__inference_GRU_2_layer_call_fn_1689485
'__inference_GRU_2_layer_call_fn_1689156
'__inference_GRU_2_layer_call_fn_1689496
'__inference_GRU_2_layer_call_fn_1689145е
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
ъ2ч
@__inference_OUT_layer_call_and_return_conditional_losses_1689527Ђ
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
Я2Ь
%__inference_OUT_layer_call_fn_1689536Ђ
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
8B6
%__inference_signature_wrapper_1686688GRU_1_input
и2е
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1689576
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1689616О
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
Ђ2
-__inference_gru_cell_82_layer_call_fn_1689644
-__inference_gru_cell_82_layer_call_fn_1689630О
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
и2е
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1689684
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1689724О
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
Ђ2
-__inference_gru_cell_83_layer_call_fn_1689738
-__inference_gru_cell_83_layer_call_fn_1689752О
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
 б
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688295" !OЂL
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
 б
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688454" !OЂL
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
 З
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688635q" !?Ђ<
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
 З
B__inference_GRU_1_layer_call_and_return_conditional_losses_1688794q" !?Ђ<
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
 Ј
'__inference_GRU_1_layer_call_fn_1688465}" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЈ
'__inference_GRU_1_layer_call_fn_1688476}" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџ
'__inference_GRU_1_layer_call_fn_1688805d" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
'__inference_GRU_1_layer_call_fn_1688816d" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЗ
B__inference_GRU_2_layer_call_and_return_conditional_losses_1688975q%#$?Ђ<
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
 З
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689134q%#$?Ђ<
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
 б
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689315%#$OЂL
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
 б
B__inference_GRU_2_layer_call_and_return_conditional_losses_1689474%#$OЂL
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
 
'__inference_GRU_2_layer_call_fn_1689145d%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
'__inference_GRU_2_layer_call_fn_1689156d%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЈ
'__inference_GRU_2_layer_call_fn_1689485}%#$OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЈ
'__inference_GRU_2_layer_call_fn_1689496}%#$OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџЈ
@__inference_OUT_layer_call_and_return_conditional_losses_1689527d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
%__inference_OUT_layer_call_fn_1689536W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџТ
G__inference_Supervisor_layer_call_and_return_conditional_losses_1687029w" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Т
G__inference_Supervisor_layer_call_and_return_conditional_losses_1687370w" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
G__inference_Supervisor_layer_call_and_return_conditional_losses_1687753r" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
G__inference_Supervisor_layer_call_and_return_conditional_losses_1688094r" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 
,__inference_Supervisor_layer_call_fn_1687391j" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
,__inference_Supervisor_layer_call_fn_1687412j" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
,__inference_Supervisor_layer_call_fn_1688115e" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
,__inference_Supervisor_layer_call_fn_1688136e" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
"__inference__wrapped_model_1684683s" !%#$8Ђ5
.Ђ+
)&
GRU_1_inputџџџџџџџџџ
Њ "-Њ*
(
OUT!
OUTџџџџџџџџџ
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1689576З" !\ЂY
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
 
H__inference_gru_cell_82_layer_call_and_return_conditional_losses_1689616З" !\ЂY
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
 л
-__inference_gru_cell_82_layer_call_fn_1689630Љ" !\ЂY
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
1/0џџџџџџџџџл
-__inference_gru_cell_82_layer_call_fn_1689644Љ" !\ЂY
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
1/0џџџџџџџџџ
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1689684З%#$\ЂY
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
 
H__inference_gru_cell_83_layer_call_and_return_conditional_losses_1689724З%#$\ЂY
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
 л
-__inference_gru_cell_83_layer_call_fn_1689738Љ%#$\ЂY
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
1/0џџџџџџџџџл
-__inference_gru_cell_83_layer_call_fn_1689752Љ%#$\ЂY
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
1/0џџџџџџџџџЌ
%__inference_signature_wrapper_1686688" !%#$GЂD
Ђ 
=Њ:
8
GRU_1_input)&
GRU_1_inputџџџџџџџџџ"-Њ*
(
OUT!
OUTџџџџџџџџџ