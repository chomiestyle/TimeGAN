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
GRU_1/gru_cell_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameGRU_1/gru_cell_54/kernel

,GRU_1/gru_cell_54/kernel/Read/ReadVariableOpReadVariableOpGRU_1/gru_cell_54/kernel*
_output_shapes

:<*
dtype0
 
"GRU_1/gru_cell_54/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*3
shared_name$"GRU_1/gru_cell_54/recurrent_kernel

6GRU_1/gru_cell_54/recurrent_kernel/Read/ReadVariableOpReadVariableOp"GRU_1/gru_cell_54/recurrent_kernel*
_output_shapes

:<*
dtype0

GRU_1/gru_cell_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameGRU_1/gru_cell_54/bias

*GRU_1/gru_cell_54/bias/Read/ReadVariableOpReadVariableOpGRU_1/gru_cell_54/bias*
_output_shapes

:<*
dtype0

GRU_2/gru_cell_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*)
shared_nameGRU_2/gru_cell_55/kernel

,GRU_2/gru_cell_55/kernel/Read/ReadVariableOpReadVariableOpGRU_2/gru_cell_55/kernel*
_output_shapes

:<*
dtype0
 
"GRU_2/gru_cell_55/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*3
shared_name$"GRU_2/gru_cell_55/recurrent_kernel

6GRU_2/gru_cell_55/recurrent_kernel/Read/ReadVariableOpReadVariableOp"GRU_2/gru_cell_55/recurrent_kernel*
_output_shapes

:<*
dtype0

GRU_2/gru_cell_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*'
shared_nameGRU_2/gru_cell_55/bias

*GRU_2/gru_cell_55/bias/Read/ReadVariableOpReadVariableOpGRU_2/gru_cell_55/bias*
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
	variables
&layer_metrics
'metrics
(layer_regularization_losses
trainable_variables
)non_trainable_variables

*layers
regularization_losses
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
	variables
/layer_metrics
0metrics
1layer_regularization_losses
trainable_variables
2non_trainable_variables

3states

4layers
regularization_losses
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
	variables
9layer_metrics
:metrics
;layer_regularization_losses
trainable_variables
<non_trainable_variables

=states

>layers
regularization_losses
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
	variables
?layer_metrics
@metrics
Alayer_regularization_losses
trainable_variables
Bnon_trainable_variables

Clayers
regularization_losses
TR
VARIABLE_VALUEGRU_1/gru_cell_54/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"GRU_1/gru_cell_54/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEGRU_1/gru_cell_54/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEGRU_2/gru_cell_55/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"GRU_2/gru_cell_55/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEGRU_2/gru_cell_55/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

0
1
2
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
+	variables
Dlayer_metrics
Emetrics
Flayer_regularization_losses
,trainable_variables
Gnon_trainable_variables

Hlayers
-regularization_losses
 
 
 
 
 

	0
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
5	variables
Ilayer_metrics
Jmetrics
Klayer_regularization_losses
6trainable_variables
Lnon_trainable_variables

Mlayers
7regularization_losses
 
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

serving_default_GRU_1_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_GRU_1_inputGRU_1/gru_cell_54/biasGRU_1/gru_cell_54/kernel"GRU_1/gru_cell_54/recurrent_kernelGRU_2/gru_cell_55/biasGRU_2/gru_cell_55/kernel"GRU_2/gru_cell_55/recurrent_kernel
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
%__inference_signature_wrapper_1126978
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameOUT/kernel/Read/ReadVariableOpOUT/bias/Read/ReadVariableOp,GRU_1/gru_cell_54/kernel/Read/ReadVariableOp6GRU_1/gru_cell_54/recurrent_kernel/Read/ReadVariableOp*GRU_1/gru_cell_54/bias/Read/ReadVariableOp,GRU_2/gru_cell_55/kernel/Read/ReadVariableOp6GRU_2/gru_cell_55/recurrent_kernel/Read/ReadVariableOp*GRU_2/gru_cell_55/bias/Read/ReadVariableOpConst*
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
 __inference__traced_save_1130089
с
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
OUT/kernelOUT/biasGRU_1/gru_cell_54/kernel"GRU_1/gru_cell_54/recurrent_kernelGRU_1/gru_cell_54/biasGRU_2/gru_cell_55/kernel"GRU_2/gru_cell_55/recurrent_kernelGRU_2/gru_cell_55/bias*
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
#__inference__traced_restore_1130123кІ&
@
Ж
while_body_1128495
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_54_readvariableop_resource_06
2while_gru_cell_54_matmul_readvariableop_resource_08
4while_gru_cell_54_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_54_readvariableop_resource4
0while_gru_cell_54_matmul_readvariableop_resource6
2while_gru_cell_54_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_54/ReadVariableOpReadVariableOp+while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_54/ReadVariableOp 
while/gru_cell_54/unstackUnpack(while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_54/unstackХ
'while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_54/MatMul/ReadVariableOpг
while/gru_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMulЛ
while/gru_cell_54/BiasAddBiasAdd"while/gru_cell_54/MatMul:product:0"while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAddt
while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_54/Const
!while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_54/split/split_dimє
while/gru_cell_54/splitSplit*while/gru_cell_54/split/split_dim:output:0"while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/splitЫ
)while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_54/MatMul_1/ReadVariableOpМ
while/gru_cell_54/MatMul_1MatMulwhile_placeholder_21while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMul_1С
while/gru_cell_54/BiasAdd_1BiasAdd$while/gru_cell_54/MatMul_1:product:0"while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAdd_1
while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_54/Const_1
#while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_54/split_1/split_dim­
while/gru_cell_54/split_1SplitV$while/gru_cell_54/BiasAdd_1:output:0"while/gru_cell_54/Const_1:output:0,while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/split_1Џ
while/gru_cell_54/addAddV2 while/gru_cell_54/split:output:0"while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add
while/gru_cell_54/SigmoidSigmoidwhile/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/SigmoidГ
while/gru_cell_54/add_1AddV2 while/gru_cell_54/split:output:1"while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_1
while/gru_cell_54/Sigmoid_1Sigmoidwhile/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Sigmoid_1Ќ
while/gru_cell_54/mulMulwhile/gru_cell_54/Sigmoid_1:y:0"while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mulЊ
while/gru_cell_54/add_2AddV2 while/gru_cell_54/split:output:2while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_2
while/gru_cell_54/TanhTanhwhile/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Tanh
while/gru_cell_54/mul_1Mulwhile/gru_cell_54/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_1w
while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_54/sub/xЈ
while/gru_cell_54/subSub while/gru_cell_54/sub/x:output:0while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/subЂ
while/gru_cell_54/mul_2Mulwhile/gru_cell_54/sub:z:0while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_2Ї
while/gru_cell_54/add_3AddV2while/gru_cell_54/mul_1:z:0while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_54/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_54_matmul_1_readvariableop_resource4while_gru_cell_54_matmul_1_readvariableop_resource_0"f
0while_gru_cell_54_matmul_readvariableop_resource2while_gru_cell_54_matmul_readvariableop_resource_0"X
)while_gru_cell_54_readvariableop_resource+while_gru_cell_54_readvariableop_resource_0")
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
GRU_2_while_body_1127543(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_55_readvariableop_resource_0<
8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_55_readvariableop_resource:
6gru_2_while_gru_cell_55_matmul_readvariableop_resource<
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_55/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_55/ReadVariableOpВ
GRU_2/while/gru_cell_55/unstackUnpack.GRU_2/while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_55/unstackз
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_55/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_55/MatMulг
GRU_2/while/gru_cell_55/BiasAddBiasAdd(GRU_2/while/gru_cell_55/MatMul:product:0(GRU_2/while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_55/BiasAdd
GRU_2/while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_55/Const
'GRU_2/while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_55/split/split_dim
GRU_2/while/gru_cell_55/splitSplit0GRU_2/while/gru_cell_55/split/split_dim:output:0(GRU_2/while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_55/splitн
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_55/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_55/MatMul_1й
!GRU_2/while/gru_cell_55/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_55/MatMul_1:product:0(GRU_2/while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_55/BiasAdd_1
GRU_2/while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_55/Const_1Ё
)GRU_2/while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_55/split_1/split_dimЫ
GRU_2/while/gru_cell_55/split_1SplitV*GRU_2/while/gru_cell_55/BiasAdd_1:output:0(GRU_2/while/gru_cell_55/Const_1:output:02GRU_2/while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_55/split_1Ч
GRU_2/while/gru_cell_55/addAddV2&GRU_2/while/gru_cell_55/split:output:0(GRU_2/while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add 
GRU_2/while/gru_cell_55/SigmoidSigmoidGRU_2/while/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_55/SigmoidЫ
GRU_2/while/gru_cell_55/add_1AddV2&GRU_2/while/gru_cell_55/split:output:1(GRU_2/while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_1І
!GRU_2/while/gru_cell_55/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_55/Sigmoid_1Ф
GRU_2/while/gru_cell_55/mulMul%GRU_2/while/gru_cell_55/Sigmoid_1:y:0(GRU_2/while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mulТ
GRU_2/while/gru_cell_55/add_2AddV2&GRU_2/while/gru_cell_55/split:output:2GRU_2/while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_2
GRU_2/while/gru_cell_55/TanhTanh!GRU_2/while/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/TanhЗ
GRU_2/while/gru_cell_55/mul_1Mul#GRU_2/while/gru_cell_55/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_1
GRU_2/while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_55/sub/xР
GRU_2/while/gru_cell_55/subSub&GRU_2/while/gru_cell_55/sub/x:output:0#GRU_2/while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/subК
GRU_2/while/gru_cell_55/mul_2MulGRU_2/while/gru_cell_55/sub:z:0 GRU_2/while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_2П
GRU_2/while/gru_cell_55/add_3AddV2!GRU_2/while/gru_cell_55/mul_1:z:0!GRU_2/while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_55/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resource:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_55_matmul_readvariableop_resource8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_55_readvariableop_resource1gru_2_while_gru_cell_55_readvariableop_resource_0"5
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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1126259

inputs'
#gru_cell_54_readvariableop_resource.
*gru_cell_54_matmul_readvariableop_resource0
,gru_cell_54_matmul_1_readvariableop_resource
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
gru_cell_54/ReadVariableOpReadVariableOp#gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_54/ReadVariableOp
gru_cell_54/unstackUnpack"gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_54/unstackБ
!gru_cell_54/MatMul/ReadVariableOpReadVariableOp*gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_54/MatMul/ReadVariableOpЉ
gru_cell_54/MatMulMatMulstrided_slice_2:output:0)gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMulЃ
gru_cell_54/BiasAddBiasAddgru_cell_54/MatMul:product:0gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAddh
gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_54/Const
gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split/split_dimм
gru_cell_54/splitSplit$gru_cell_54/split/split_dim:output:0gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/splitЗ
#gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_54/MatMul_1/ReadVariableOpЅ
gru_cell_54/MatMul_1MatMulzeros:output:0+gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMul_1Љ
gru_cell_54/BiasAdd_1BiasAddgru_cell_54/MatMul_1:product:0gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAdd_1
gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_54/Const_1
gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split_1/split_dim
gru_cell_54/split_1SplitVgru_cell_54/BiasAdd_1:output:0gru_cell_54/Const_1:output:0&gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/split_1
gru_cell_54/addAddV2gru_cell_54/split:output:0gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add|
gru_cell_54/SigmoidSigmoidgru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid
gru_cell_54/add_1AddV2gru_cell_54/split:output:1gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_1
gru_cell_54/Sigmoid_1Sigmoidgru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid_1
gru_cell_54/mulMulgru_cell_54/Sigmoid_1:y:0gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul
gru_cell_54/add_2AddV2gru_cell_54/split:output:2gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_2u
gru_cell_54/TanhTanhgru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Tanh
gru_cell_54/mul_1Mulgru_cell_54/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_1k
gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_54/sub/x
gru_cell_54/subSubgru_cell_54/sub/x:output:0gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/sub
gru_cell_54/mul_2Mulgru_cell_54/sub:z:0gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_2
gru_cell_54/add_3AddV2gru_cell_54/mul_1:z:0gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_54_readvariableop_resource*gru_cell_54_matmul_readvariableop_resource,gru_cell_54_matmul_1_readvariableop_resource*
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
while_body_1126169*
condR
while_cond_1126168*8
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


'__inference_GRU_2_layer_call_fn_1129786

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
B__inference_GRU_2_layer_call_and_return_conditional_losses_11267652
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
&
ч
#__inference__traced_restore_1130123
file_prefix
assignvariableop_out_kernel
assignvariableop_1_out_bias/
+assignvariableop_2_gru_1_gru_cell_54_kernel9
5assignvariableop_3_gru_1_gru_cell_54_recurrent_kernel-
)assignvariableop_4_gru_1_gru_cell_54_bias/
+assignvariableop_5_gru_2_gru_cell_55_kernel9
5assignvariableop_6_gru_2_gru_cell_55_recurrent_kernel-
)assignvariableop_7_gru_2_gru_cell_55_bias

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
AssignVariableOp_2AssignVariableOp+assignvariableop_2_gru_1_gru_cell_54_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3К
AssignVariableOp_3AssignVariableOp5assignvariableop_3_gru_1_gru_cell_54_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ў
AssignVariableOp_4AssignVariableOp)assignvariableop_4_gru_1_gru_cell_54_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5А
AssignVariableOp_5AssignVariableOp+assignvariableop_5_gru_2_gru_cell_55_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6К
AssignVariableOp_6AssignVariableOp5assignvariableop_6_gru_2_gru_cell_55_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ў
AssignVariableOp_7AssignVariableOp)assignvariableop_7_gru_2_gru_cell_55_biasIdentity_7:output:0"/device:CPU:0*
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
while_body_1128654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_54_readvariableop_resource_06
2while_gru_cell_54_matmul_readvariableop_resource_08
4while_gru_cell_54_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_54_readvariableop_resource4
0while_gru_cell_54_matmul_readvariableop_resource6
2while_gru_cell_54_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_54/ReadVariableOpReadVariableOp+while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_54/ReadVariableOp 
while/gru_cell_54/unstackUnpack(while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_54/unstackХ
'while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_54/MatMul/ReadVariableOpг
while/gru_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMulЛ
while/gru_cell_54/BiasAddBiasAdd"while/gru_cell_54/MatMul:product:0"while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAddt
while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_54/Const
!while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_54/split/split_dimє
while/gru_cell_54/splitSplit*while/gru_cell_54/split/split_dim:output:0"while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/splitЫ
)while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_54/MatMul_1/ReadVariableOpМ
while/gru_cell_54/MatMul_1MatMulwhile_placeholder_21while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMul_1С
while/gru_cell_54/BiasAdd_1BiasAdd$while/gru_cell_54/MatMul_1:product:0"while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAdd_1
while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_54/Const_1
#while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_54/split_1/split_dim­
while/gru_cell_54/split_1SplitV$while/gru_cell_54/BiasAdd_1:output:0"while/gru_cell_54/Const_1:output:0,while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/split_1Џ
while/gru_cell_54/addAddV2 while/gru_cell_54/split:output:0"while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add
while/gru_cell_54/SigmoidSigmoidwhile/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/SigmoidГ
while/gru_cell_54/add_1AddV2 while/gru_cell_54/split:output:1"while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_1
while/gru_cell_54/Sigmoid_1Sigmoidwhile/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Sigmoid_1Ќ
while/gru_cell_54/mulMulwhile/gru_cell_54/Sigmoid_1:y:0"while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mulЊ
while/gru_cell_54/add_2AddV2 while/gru_cell_54/split:output:2while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_2
while/gru_cell_54/TanhTanhwhile/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Tanh
while/gru_cell_54/mul_1Mulwhile/gru_cell_54/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_1w
while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_54/sub/xЈ
while/gru_cell_54/subSub while/gru_cell_54/sub/x:output:0while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/subЂ
while/gru_cell_54/mul_2Mulwhile/gru_cell_54/sub:z:0while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_2Ї
while/gru_cell_54/add_3AddV2while/gru_cell_54/mul_1:z:0while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_54/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_54_matmul_1_readvariableop_resource4while_gru_cell_54_matmul_1_readvariableop_resource_0"f
0while_gru_cell_54_matmul_readvariableop_resource2while_gru_cell_54_matmul_readvariableop_resource_0"X
)while_gru_cell_54_readvariableop_resource+while_gru_cell_54_readvariableop_resource_0")
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
џ!
т
while_body_1125462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_54_1125484_0
while_gru_cell_54_1125486_0
while_gru_cell_54_1125488_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_54_1125484
while_gru_cell_54_1125486
while_gru_cell_54_1125488Ђ)while/gru_cell_54/StatefulPartitionedCallУ
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
)while/gru_cell_54/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_54_1125484_0while_gru_cell_54_1125486_0while_gru_cell_54_1125488_0*
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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_11250852+
)while/gru_cell_54/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_54/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_54/StatefulPartitionedCall:output:1*^while/gru_cell_54/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_54_1125484while_gru_cell_54_1125484_0"8
while_gru_cell_54_1125486while_gru_cell_54_1125486_0"8
while_gru_cell_54_1125488while_gru_cell_54_1125488_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_54/StatefulPartitionedCall)while/gru_cell_54/StatefulPartitionedCall: 
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_1126936

inputs
gru_1_1126916
gru_1_1126918
gru_1_1126920
gru_2_1126923
gru_2_1126925
gru_2_1126927
out_1126930
out_1126932
identityЂGRU_1/StatefulPartitionedCallЂGRU_2/StatefulPartitionedCallЂOUT/StatefulPartitionedCall
GRU_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_1126916gru_1_1126918gru_1_1126920*
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_11264182
GRU_1/StatefulPartitionedCallН
GRU_2/StatefulPartitionedCallStatefulPartitionedCall&GRU_1/StatefulPartitionedCall:output:0gru_2_1126923gru_2_1126925gru_2_1126927*
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_11267652
GRU_2/StatefulPartitionedCallЂ
OUT/StatefulPartitionedCallStatefulPartitionedCall&GRU_2/StatefulPartitionedCall:output:0out_1126930out_1126932*
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
@__inference_OUT_layer_call_and_return_conditional_losses_11268262
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
е
Џ
while_cond_1128494
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1128494___redundant_placeholder05
1while_while_cond_1128494___redundant_placeholder15
1while_while_cond_1128494___redundant_placeholder25
1while_while_cond_1128494___redundant_placeholder3
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
while_cond_1125461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1125461___redundant_placeholder05
1while_while_cond_1125461___redundant_placeholder15
1while_while_cond_1125461___redundant_placeholder25
1while_while_cond_1125461___redundant_placeholder3
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
Ћ
@__inference_OUT_layer_call_and_return_conditional_losses_1126826

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
ЊX
і
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129424
inputs_0'
#gru_cell_55_readvariableop_resource.
*gru_cell_55_matmul_readvariableop_resource0
,gru_cell_55_matmul_1_readvariableop_resource
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
gru_cell_55/ReadVariableOpReadVariableOp#gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_55/ReadVariableOp
gru_cell_55/unstackUnpack"gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_55/unstackБ
!gru_cell_55/MatMul/ReadVariableOpReadVariableOp*gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_55/MatMul/ReadVariableOpЉ
gru_cell_55/MatMulMatMulstrided_slice_2:output:0)gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMulЃ
gru_cell_55/BiasAddBiasAddgru_cell_55/MatMul:product:0gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAddh
gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_55/Const
gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split/split_dimм
gru_cell_55/splitSplit$gru_cell_55/split/split_dim:output:0gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/splitЗ
#gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_55/MatMul_1/ReadVariableOpЅ
gru_cell_55/MatMul_1MatMulzeros:output:0+gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMul_1Љ
gru_cell_55/BiasAdd_1BiasAddgru_cell_55/MatMul_1:product:0gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAdd_1
gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_55/Const_1
gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split_1/split_dim
gru_cell_55/split_1SplitVgru_cell_55/BiasAdd_1:output:0gru_cell_55/Const_1:output:0&gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/split_1
gru_cell_55/addAddV2gru_cell_55/split:output:0gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add|
gru_cell_55/SigmoidSigmoidgru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid
gru_cell_55/add_1AddV2gru_cell_55/split:output:1gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_1
gru_cell_55/Sigmoid_1Sigmoidgru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid_1
gru_cell_55/mulMulgru_cell_55/Sigmoid_1:y:0gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul
gru_cell_55/add_2AddV2gru_cell_55/split:output:2gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_2u
gru_cell_55/TanhTanhgru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Tanh
gru_cell_55/mul_1Mulgru_cell_55/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_1k
gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_55/sub/x
gru_cell_55/subSubgru_cell_55/sub/x:output:0gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/sub
gru_cell_55/mul_2Mulgru_cell_55/sub:z:0gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_2
gru_cell_55/add_3AddV2gru_cell_55/mul_1:z:0gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_55_readvariableop_resource*gru_cell_55_matmul_readvariableop_resource,gru_cell_55_matmul_1_readvariableop_resource*
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
while_body_1129334*
condR
while_cond_1129333*8
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
О
Ї
 __inference__traced_save_1130089
file_prefix)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop7
3savev2_gru_1_gru_cell_54_kernel_read_readvariableopA
=savev2_gru_1_gru_cell_54_recurrent_kernel_read_readvariableop5
1savev2_gru_1_gru_cell_54_bias_read_readvariableop7
3savev2_gru_2_gru_cell_55_kernel_read_readvariableopA
=savev2_gru_2_gru_cell_55_recurrent_kernel_read_readvariableop5
1savev2_gru_2_gru_cell_55_bias_read_readvariableop
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
value3B1 B+_temp_ad48c2c50852484d96afdf2b68ef3759/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop3savev2_gru_1_gru_cell_54_kernel_read_readvariableop=savev2_gru_1_gru_cell_54_recurrent_kernel_read_readvariableop1savev2_gru_1_gru_cell_54_bias_read_readvariableop3savev2_gru_2_gru_cell_55_kernel_read_readvariableop=savev2_gru_2_gru_cell_55_recurrent_kernel_read_readvariableop1savev2_gru_2_gru_cell_55_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
Яс

G__inference_Supervisor_layer_call_and_return_conditional_losses_1128043

inputs-
)gru_1_gru_cell_54_readvariableop_resource4
0gru_1_gru_cell_54_matmul_readvariableop_resource6
2gru_1_gru_cell_54_matmul_1_readvariableop_resource-
)gru_2_gru_cell_55_readvariableop_resource4
0gru_2_gru_cell_55_matmul_readvariableop_resource6
2gru_2_gru_cell_55_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_54/ReadVariableOpReadVariableOp)gru_1_gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_54/ReadVariableOp 
GRU_1/gru_cell_54/unstackUnpack(GRU_1/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_54/unstackУ
'GRU_1/gru_cell_54/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_54/MatMul/ReadVariableOpС
GRU_1/gru_cell_54/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMulЛ
GRU_1/gru_cell_54/BiasAddBiasAdd"GRU_1/gru_cell_54/MatMul:product:0"GRU_1/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAddt
GRU_1/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_54/Const
!GRU_1/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_54/split/split_dimє
GRU_1/gru_cell_54/splitSplit*GRU_1/gru_cell_54/split/split_dim:output:0"GRU_1/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/splitЩ
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_54/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMul_1С
GRU_1/gru_cell_54/BiasAdd_1BiasAdd$GRU_1/gru_cell_54/MatMul_1:product:0"GRU_1/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAdd_1
GRU_1/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_54/Const_1
#GRU_1/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_54/split_1/split_dim­
GRU_1/gru_cell_54/split_1SplitV$GRU_1/gru_cell_54/BiasAdd_1:output:0"GRU_1/gru_cell_54/Const_1:output:0,GRU_1/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/split_1Џ
GRU_1/gru_cell_54/addAddV2 GRU_1/gru_cell_54/split:output:0"GRU_1/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add
GRU_1/gru_cell_54/SigmoidSigmoidGRU_1/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/SigmoidГ
GRU_1/gru_cell_54/add_1AddV2 GRU_1/gru_cell_54/split:output:1"GRU_1/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_1
GRU_1/gru_cell_54/Sigmoid_1SigmoidGRU_1/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Sigmoid_1Ќ
GRU_1/gru_cell_54/mulMulGRU_1/gru_cell_54/Sigmoid_1:y:0"GRU_1/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mulЊ
GRU_1/gru_cell_54/add_2AddV2 GRU_1/gru_cell_54/split:output:2GRU_1/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_2
GRU_1/gru_cell_54/TanhTanhGRU_1/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Tanh 
GRU_1/gru_cell_54/mul_1MulGRU_1/gru_cell_54/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_1w
GRU_1/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_54/sub/xЈ
GRU_1/gru_cell_54/subSub GRU_1/gru_cell_54/sub/x:output:0GRU_1/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/subЂ
GRU_1/gru_cell_54/mul_2MulGRU_1/gru_cell_54/sub:z:0GRU_1/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_2Ї
GRU_1/gru_cell_54/add_3AddV2GRU_1/gru_cell_54/mul_1:z:0GRU_1/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_3
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
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_54_readvariableop_resource0gru_1_gru_cell_54_matmul_readvariableop_resource2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
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
GRU_1_while_body_1127771*$
condR
GRU_1_while_cond_1127770*8
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
 GRU_2/gru_cell_55/ReadVariableOpReadVariableOp)gru_2_gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_55/ReadVariableOp 
GRU_2/gru_cell_55/unstackUnpack(GRU_2/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_55/unstackУ
'GRU_2/gru_cell_55/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_55/MatMul/ReadVariableOpС
GRU_2/gru_cell_55/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMulЛ
GRU_2/gru_cell_55/BiasAddBiasAdd"GRU_2/gru_cell_55/MatMul:product:0"GRU_2/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAddt
GRU_2/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_55/Const
!GRU_2/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_55/split/split_dimє
GRU_2/gru_cell_55/splitSplit*GRU_2/gru_cell_55/split/split_dim:output:0"GRU_2/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/splitЩ
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_55/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMul_1С
GRU_2/gru_cell_55/BiasAdd_1BiasAdd$GRU_2/gru_cell_55/MatMul_1:product:0"GRU_2/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAdd_1
GRU_2/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_55/Const_1
#GRU_2/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_55/split_1/split_dim­
GRU_2/gru_cell_55/split_1SplitV$GRU_2/gru_cell_55/BiasAdd_1:output:0"GRU_2/gru_cell_55/Const_1:output:0,GRU_2/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/split_1Џ
GRU_2/gru_cell_55/addAddV2 GRU_2/gru_cell_55/split:output:0"GRU_2/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add
GRU_2/gru_cell_55/SigmoidSigmoidGRU_2/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/SigmoidГ
GRU_2/gru_cell_55/add_1AddV2 GRU_2/gru_cell_55/split:output:1"GRU_2/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_1
GRU_2/gru_cell_55/Sigmoid_1SigmoidGRU_2/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Sigmoid_1Ќ
GRU_2/gru_cell_55/mulMulGRU_2/gru_cell_55/Sigmoid_1:y:0"GRU_2/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mulЊ
GRU_2/gru_cell_55/add_2AddV2 GRU_2/gru_cell_55/split:output:2GRU_2/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_2
GRU_2/gru_cell_55/TanhTanhGRU_2/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Tanh 
GRU_2/gru_cell_55/mul_1MulGRU_2/gru_cell_55/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_1w
GRU_2/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_55/sub/xЈ
GRU_2/gru_cell_55/subSub GRU_2/gru_cell_55/sub/x:output:0GRU_2/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/subЂ
GRU_2/gru_cell_55/mul_2MulGRU_2/gru_cell_55/sub:z:0GRU_2/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_2Ї
GRU_2/gru_cell_55/add_3AddV2GRU_2/gru_cell_55/mul_1:z:0GRU_2/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_3
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
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_55_readvariableop_resource0gru_2_gru_cell_55_matmul_readvariableop_resource2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
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
GRU_2_while_body_1127926*$
condR
GRU_2_while_cond_1127925*8
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
Т
р
,__inference_Supervisor_layer_call_fn_1127681
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_11268922
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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128585

inputs'
#gru_cell_54_readvariableop_resource.
*gru_cell_54_matmul_readvariableop_resource0
,gru_cell_54_matmul_1_readvariableop_resource
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
gru_cell_54/ReadVariableOpReadVariableOp#gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_54/ReadVariableOp
gru_cell_54/unstackUnpack"gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_54/unstackБ
!gru_cell_54/MatMul/ReadVariableOpReadVariableOp*gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_54/MatMul/ReadVariableOpЉ
gru_cell_54/MatMulMatMulstrided_slice_2:output:0)gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMulЃ
gru_cell_54/BiasAddBiasAddgru_cell_54/MatMul:product:0gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAddh
gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_54/Const
gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split/split_dimм
gru_cell_54/splitSplit$gru_cell_54/split/split_dim:output:0gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/splitЗ
#gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_54/MatMul_1/ReadVariableOpЅ
gru_cell_54/MatMul_1MatMulzeros:output:0+gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMul_1Љ
gru_cell_54/BiasAdd_1BiasAddgru_cell_54/MatMul_1:product:0gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAdd_1
gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_54/Const_1
gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split_1/split_dim
gru_cell_54/split_1SplitVgru_cell_54/BiasAdd_1:output:0gru_cell_54/Const_1:output:0&gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/split_1
gru_cell_54/addAddV2gru_cell_54/split:output:0gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add|
gru_cell_54/SigmoidSigmoidgru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid
gru_cell_54/add_1AddV2gru_cell_54/split:output:1gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_1
gru_cell_54/Sigmoid_1Sigmoidgru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid_1
gru_cell_54/mulMulgru_cell_54/Sigmoid_1:y:0gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul
gru_cell_54/add_2AddV2gru_cell_54/split:output:2gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_2u
gru_cell_54/TanhTanhgru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Tanh
gru_cell_54/mul_1Mulgru_cell_54/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_1k
gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_54/sub/x
gru_cell_54/subSubgru_cell_54/sub/x:output:0gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/sub
gru_cell_54/mul_2Mulgru_cell_54/sub:z:0gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_2
gru_cell_54/add_3AddV2gru_cell_54/mul_1:z:0gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_54_readvariableop_resource*gru_cell_54_matmul_readvariableop_resource,gru_cell_54_matmul_1_readvariableop_resource*
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
while_body_1128495*
condR
while_cond_1128494*8
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128925
inputs_0'
#gru_cell_54_readvariableop_resource.
*gru_cell_54_matmul_readvariableop_resource0
,gru_cell_54_matmul_1_readvariableop_resource
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
gru_cell_54/ReadVariableOpReadVariableOp#gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_54/ReadVariableOp
gru_cell_54/unstackUnpack"gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_54/unstackБ
!gru_cell_54/MatMul/ReadVariableOpReadVariableOp*gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_54/MatMul/ReadVariableOpЉ
gru_cell_54/MatMulMatMulstrided_slice_2:output:0)gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMulЃ
gru_cell_54/BiasAddBiasAddgru_cell_54/MatMul:product:0gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAddh
gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_54/Const
gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split/split_dimм
gru_cell_54/splitSplit$gru_cell_54/split/split_dim:output:0gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/splitЗ
#gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_54/MatMul_1/ReadVariableOpЅ
gru_cell_54/MatMul_1MatMulzeros:output:0+gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMul_1Љ
gru_cell_54/BiasAdd_1BiasAddgru_cell_54/MatMul_1:product:0gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAdd_1
gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_54/Const_1
gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split_1/split_dim
gru_cell_54/split_1SplitVgru_cell_54/BiasAdd_1:output:0gru_cell_54/Const_1:output:0&gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/split_1
gru_cell_54/addAddV2gru_cell_54/split:output:0gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add|
gru_cell_54/SigmoidSigmoidgru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid
gru_cell_54/add_1AddV2gru_cell_54/split:output:1gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_1
gru_cell_54/Sigmoid_1Sigmoidgru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid_1
gru_cell_54/mulMulgru_cell_54/Sigmoid_1:y:0gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul
gru_cell_54/add_2AddV2gru_cell_54/split:output:2gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_2u
gru_cell_54/TanhTanhgru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Tanh
gru_cell_54/mul_1Mulgru_cell_54/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_1k
gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_54/sub/x
gru_cell_54/subSubgru_cell_54/sub/x:output:0gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/sub
gru_cell_54/mul_2Mulgru_cell_54/sub:z:0gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_2
gru_cell_54/add_3AddV2gru_cell_54/mul_1:z:0gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_54_readvariableop_resource*gru_cell_54_matmul_readvariableop_resource,gru_cell_54_matmul_1_readvariableop_resource*
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
while_body_1128835*
condR
while_cond_1128834*8
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
е
Џ
while_cond_1126515
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1126515___redundant_placeholder05
1while_while_cond_1126515___redundant_placeholder15
1while_while_cond_1126515___redundant_placeholder25
1while_while_cond_1126515___redundant_placeholder3
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_1127319
gru_1_input-
)gru_1_gru_cell_54_readvariableop_resource4
0gru_1_gru_cell_54_matmul_readvariableop_resource6
2gru_1_gru_cell_54_matmul_1_readvariableop_resource-
)gru_2_gru_cell_55_readvariableop_resource4
0gru_2_gru_cell_55_matmul_readvariableop_resource6
2gru_2_gru_cell_55_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_54/ReadVariableOpReadVariableOp)gru_1_gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_54/ReadVariableOp 
GRU_1/gru_cell_54/unstackUnpack(GRU_1/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_54/unstackУ
'GRU_1/gru_cell_54/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_54/MatMul/ReadVariableOpС
GRU_1/gru_cell_54/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMulЛ
GRU_1/gru_cell_54/BiasAddBiasAdd"GRU_1/gru_cell_54/MatMul:product:0"GRU_1/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAddt
GRU_1/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_54/Const
!GRU_1/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_54/split/split_dimє
GRU_1/gru_cell_54/splitSplit*GRU_1/gru_cell_54/split/split_dim:output:0"GRU_1/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/splitЩ
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_54/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMul_1С
GRU_1/gru_cell_54/BiasAdd_1BiasAdd$GRU_1/gru_cell_54/MatMul_1:product:0"GRU_1/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAdd_1
GRU_1/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_54/Const_1
#GRU_1/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_54/split_1/split_dim­
GRU_1/gru_cell_54/split_1SplitV$GRU_1/gru_cell_54/BiasAdd_1:output:0"GRU_1/gru_cell_54/Const_1:output:0,GRU_1/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/split_1Џ
GRU_1/gru_cell_54/addAddV2 GRU_1/gru_cell_54/split:output:0"GRU_1/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add
GRU_1/gru_cell_54/SigmoidSigmoidGRU_1/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/SigmoidГ
GRU_1/gru_cell_54/add_1AddV2 GRU_1/gru_cell_54/split:output:1"GRU_1/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_1
GRU_1/gru_cell_54/Sigmoid_1SigmoidGRU_1/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Sigmoid_1Ќ
GRU_1/gru_cell_54/mulMulGRU_1/gru_cell_54/Sigmoid_1:y:0"GRU_1/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mulЊ
GRU_1/gru_cell_54/add_2AddV2 GRU_1/gru_cell_54/split:output:2GRU_1/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_2
GRU_1/gru_cell_54/TanhTanhGRU_1/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Tanh 
GRU_1/gru_cell_54/mul_1MulGRU_1/gru_cell_54/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_1w
GRU_1/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_54/sub/xЈ
GRU_1/gru_cell_54/subSub GRU_1/gru_cell_54/sub/x:output:0GRU_1/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/subЂ
GRU_1/gru_cell_54/mul_2MulGRU_1/gru_cell_54/sub:z:0GRU_1/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_2Ї
GRU_1/gru_cell_54/add_3AddV2GRU_1/gru_cell_54/mul_1:z:0GRU_1/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_3
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
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_54_readvariableop_resource0gru_1_gru_cell_54_matmul_readvariableop_resource2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
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
GRU_1_while_body_1127047*$
condR
GRU_1_while_cond_1127046*8
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
 GRU_2/gru_cell_55/ReadVariableOpReadVariableOp)gru_2_gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_55/ReadVariableOp 
GRU_2/gru_cell_55/unstackUnpack(GRU_2/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_55/unstackУ
'GRU_2/gru_cell_55/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_55/MatMul/ReadVariableOpС
GRU_2/gru_cell_55/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMulЛ
GRU_2/gru_cell_55/BiasAddBiasAdd"GRU_2/gru_cell_55/MatMul:product:0"GRU_2/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAddt
GRU_2/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_55/Const
!GRU_2/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_55/split/split_dimє
GRU_2/gru_cell_55/splitSplit*GRU_2/gru_cell_55/split/split_dim:output:0"GRU_2/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/splitЩ
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_55/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMul_1С
GRU_2/gru_cell_55/BiasAdd_1BiasAdd$GRU_2/gru_cell_55/MatMul_1:product:0"GRU_2/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAdd_1
GRU_2/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_55/Const_1
#GRU_2/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_55/split_1/split_dim­
GRU_2/gru_cell_55/split_1SplitV$GRU_2/gru_cell_55/BiasAdd_1:output:0"GRU_2/gru_cell_55/Const_1:output:0,GRU_2/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/split_1Џ
GRU_2/gru_cell_55/addAddV2 GRU_2/gru_cell_55/split:output:0"GRU_2/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add
GRU_2/gru_cell_55/SigmoidSigmoidGRU_2/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/SigmoidГ
GRU_2/gru_cell_55/add_1AddV2 GRU_2/gru_cell_55/split:output:1"GRU_2/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_1
GRU_2/gru_cell_55/Sigmoid_1SigmoidGRU_2/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Sigmoid_1Ќ
GRU_2/gru_cell_55/mulMulGRU_2/gru_cell_55/Sigmoid_1:y:0"GRU_2/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mulЊ
GRU_2/gru_cell_55/add_2AddV2 GRU_2/gru_cell_55/split:output:2GRU_2/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_2
GRU_2/gru_cell_55/TanhTanhGRU_2/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Tanh 
GRU_2/gru_cell_55/mul_1MulGRU_2/gru_cell_55/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_1w
GRU_2/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_55/sub/xЈ
GRU_2/gru_cell_55/subSub GRU_2/gru_cell_55/sub/x:output:0GRU_2/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/subЂ
GRU_2/gru_cell_55/mul_2MulGRU_2/gru_cell_55/sub:z:0GRU_2/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_2Ї
GRU_2/gru_cell_55/add_3AddV2GRU_2/gru_cell_55/mul_1:z:0GRU_2/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_3
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
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_55_readvariableop_resource0gru_2_gru_cell_55_matmul_readvariableop_resource2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
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
GRU_2_while_body_1127202*$
condR
GRU_2_while_cond_1127201*8
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
ЊX
і
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129265
inputs_0'
#gru_cell_55_readvariableop_resource.
*gru_cell_55_matmul_readvariableop_resource0
,gru_cell_55_matmul_1_readvariableop_resource
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
gru_cell_55/ReadVariableOpReadVariableOp#gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_55/ReadVariableOp
gru_cell_55/unstackUnpack"gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_55/unstackБ
!gru_cell_55/MatMul/ReadVariableOpReadVariableOp*gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_55/MatMul/ReadVariableOpЉ
gru_cell_55/MatMulMatMulstrided_slice_2:output:0)gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMulЃ
gru_cell_55/BiasAddBiasAddgru_cell_55/MatMul:product:0gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAddh
gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_55/Const
gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split/split_dimм
gru_cell_55/splitSplit$gru_cell_55/split/split_dim:output:0gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/splitЗ
#gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_55/MatMul_1/ReadVariableOpЅ
gru_cell_55/MatMul_1MatMulzeros:output:0+gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMul_1Љ
gru_cell_55/BiasAdd_1BiasAddgru_cell_55/MatMul_1:product:0gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAdd_1
gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_55/Const_1
gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split_1/split_dim
gru_cell_55/split_1SplitVgru_cell_55/BiasAdd_1:output:0gru_cell_55/Const_1:output:0&gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/split_1
gru_cell_55/addAddV2gru_cell_55/split:output:0gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add|
gru_cell_55/SigmoidSigmoidgru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid
gru_cell_55/add_1AddV2gru_cell_55/split:output:1gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_1
gru_cell_55/Sigmoid_1Sigmoidgru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid_1
gru_cell_55/mulMulgru_cell_55/Sigmoid_1:y:0gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul
gru_cell_55/add_2AddV2gru_cell_55/split:output:2gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_2u
gru_cell_55/TanhTanhgru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Tanh
gru_cell_55/mul_1Mulgru_cell_55/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_1k
gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_55/sub/x
gru_cell_55/subSubgru_cell_55/sub/x:output:0gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/sub
gru_cell_55/mul_2Mulgru_cell_55/sub:z:0gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_2
gru_cell_55/add_3AddV2gru_cell_55/mul_1:z:0gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_55_readvariableop_resource*gru_cell_55_matmul_readvariableop_resource,gru_cell_55_matmul_1_readvariableop_resource*
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
while_body_1129175*
condR
while_cond_1129174*8
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
о
Ћ
@__inference_OUT_layer_call_and_return_conditional_losses_1129817

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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1129866

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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1125970

inputs
gru_cell_55_1125894
gru_cell_55_1125896
gru_cell_55_1125898
identityЂ#gru_cell_55/StatefulPartitionedCallЂwhileD
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
#gru_cell_55/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_55_1125894gru_cell_55_1125896gru_cell_55_1125898*
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_11256072%
#gru_cell_55/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_55_1125894gru_cell_55_1125896gru_cell_55_1125898*
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
while_body_1125906*
condR
while_cond_1125905*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_55/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_55/StatefulPartitionedCall#gru_cell_55/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


'__inference_GRU_1_layer_call_fn_1128766

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
B__inference_GRU_1_layer_call_and_return_conditional_losses_11264182
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
	
Ё
GRU_1_while_cond_1127770(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1127770___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1127770___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1127770___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1127770___redundant_placeholder3
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
GRU_2_while_body_1127926(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_55_readvariableop_resource_0<
8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_55_readvariableop_resource:
6gru_2_while_gru_cell_55_matmul_readvariableop_resource<
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_55/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_55/ReadVariableOpВ
GRU_2/while/gru_cell_55/unstackUnpack.GRU_2/while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_55/unstackз
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_55/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_55/MatMulг
GRU_2/while/gru_cell_55/BiasAddBiasAdd(GRU_2/while/gru_cell_55/MatMul:product:0(GRU_2/while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_55/BiasAdd
GRU_2/while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_55/Const
'GRU_2/while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_55/split/split_dim
GRU_2/while/gru_cell_55/splitSplit0GRU_2/while/gru_cell_55/split/split_dim:output:0(GRU_2/while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_55/splitн
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_55/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_55/MatMul_1й
!GRU_2/while/gru_cell_55/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_55/MatMul_1:product:0(GRU_2/while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_55/BiasAdd_1
GRU_2/while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_55/Const_1Ё
)GRU_2/while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_55/split_1/split_dimЫ
GRU_2/while/gru_cell_55/split_1SplitV*GRU_2/while/gru_cell_55/BiasAdd_1:output:0(GRU_2/while/gru_cell_55/Const_1:output:02GRU_2/while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_55/split_1Ч
GRU_2/while/gru_cell_55/addAddV2&GRU_2/while/gru_cell_55/split:output:0(GRU_2/while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add 
GRU_2/while/gru_cell_55/SigmoidSigmoidGRU_2/while/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_55/SigmoidЫ
GRU_2/while/gru_cell_55/add_1AddV2&GRU_2/while/gru_cell_55/split:output:1(GRU_2/while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_1І
!GRU_2/while/gru_cell_55/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_55/Sigmoid_1Ф
GRU_2/while/gru_cell_55/mulMul%GRU_2/while/gru_cell_55/Sigmoid_1:y:0(GRU_2/while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mulТ
GRU_2/while/gru_cell_55/add_2AddV2&GRU_2/while/gru_cell_55/split:output:2GRU_2/while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_2
GRU_2/while/gru_cell_55/TanhTanh!GRU_2/while/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/TanhЗ
GRU_2/while/gru_cell_55/mul_1Mul#GRU_2/while/gru_cell_55/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_1
GRU_2/while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_55/sub/xР
GRU_2/while/gru_cell_55/subSub&GRU_2/while/gru_cell_55/sub/x:output:0#GRU_2/while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/subК
GRU_2/while/gru_cell_55/mul_2MulGRU_2/while/gru_cell_55/sub:z:0 GRU_2/while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_2П
GRU_2/while/gru_cell_55/add_3AddV2!GRU_2/while/gru_cell_55/mul_1:z:0!GRU_2/while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_55/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resource:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_55_matmul_readvariableop_resource8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_55_readvariableop_resource1gru_2_while_gru_cell_55_readvariableop_resource_0"5
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
Ф
ђ
#Supervisor_GRU_2_while_cond_1124855>
:supervisor_gru_2_while_supervisor_gru_2_while_loop_counterD
@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations&
"supervisor_gru_2_while_placeholder(
$supervisor_gru_2_while_placeholder_1(
$supervisor_gru_2_while_placeholder_2@
<supervisor_gru_2_while_less_supervisor_gru_2_strided_slice_1W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1124855___redundant_placeholder0W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1124855___redundant_placeholder1W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1124855___redundant_placeholder2W
Ssupervisor_gru_2_while_supervisor_gru_2_while_cond_1124855___redundant_placeholder3#
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
@
Ж
while_body_1129674
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_55_readvariableop_resource_06
2while_gru_cell_55_matmul_readvariableop_resource_08
4while_gru_cell_55_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_55_readvariableop_resource4
0while_gru_cell_55_matmul_readvariableop_resource6
2while_gru_cell_55_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_55/ReadVariableOpReadVariableOp+while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_55/ReadVariableOp 
while/gru_cell_55/unstackUnpack(while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_55/unstackХ
'while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_55/MatMul/ReadVariableOpг
while/gru_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMulЛ
while/gru_cell_55/BiasAddBiasAdd"while/gru_cell_55/MatMul:product:0"while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAddt
while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_55/Const
!while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_55/split/split_dimє
while/gru_cell_55/splitSplit*while/gru_cell_55/split/split_dim:output:0"while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/splitЫ
)while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_55/MatMul_1/ReadVariableOpМ
while/gru_cell_55/MatMul_1MatMulwhile_placeholder_21while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMul_1С
while/gru_cell_55/BiasAdd_1BiasAdd$while/gru_cell_55/MatMul_1:product:0"while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAdd_1
while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_55/Const_1
#while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_55/split_1/split_dim­
while/gru_cell_55/split_1SplitV$while/gru_cell_55/BiasAdd_1:output:0"while/gru_cell_55/Const_1:output:0,while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/split_1Џ
while/gru_cell_55/addAddV2 while/gru_cell_55/split:output:0"while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add
while/gru_cell_55/SigmoidSigmoidwhile/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/SigmoidГ
while/gru_cell_55/add_1AddV2 while/gru_cell_55/split:output:1"while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_1
while/gru_cell_55/Sigmoid_1Sigmoidwhile/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Sigmoid_1Ќ
while/gru_cell_55/mulMulwhile/gru_cell_55/Sigmoid_1:y:0"while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mulЊ
while/gru_cell_55/add_2AddV2 while/gru_cell_55/split:output:2while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_2
while/gru_cell_55/TanhTanhwhile/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Tanh
while/gru_cell_55/mul_1Mulwhile/gru_cell_55/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_1w
while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_55/sub/xЈ
while/gru_cell_55/subSub while/gru_cell_55/sub/x:output:0while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/subЂ
while/gru_cell_55/mul_2Mulwhile/gru_cell_55/sub:z:0while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_2Ї
while/gru_cell_55/add_3AddV2while/gru_cell_55/mul_1:z:0while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_55/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_55_matmul_1_readvariableop_resource4while_gru_cell_55_matmul_1_readvariableop_resource_0"f
0while_gru_cell_55_matmul_readvariableop_resource2while_gru_cell_55_matmul_readvariableop_resource_0"X
)while_gru_cell_55_readvariableop_resource+while_gru_cell_55_readvariableop_resource_0")
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
ч
ъ
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1125647

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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1126765

inputs'
#gru_cell_55_readvariableop_resource.
*gru_cell_55_matmul_readvariableop_resource0
,gru_cell_55_matmul_1_readvariableop_resource
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
gru_cell_55/ReadVariableOpReadVariableOp#gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_55/ReadVariableOp
gru_cell_55/unstackUnpack"gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_55/unstackБ
!gru_cell_55/MatMul/ReadVariableOpReadVariableOp*gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_55/MatMul/ReadVariableOpЉ
gru_cell_55/MatMulMatMulstrided_slice_2:output:0)gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMulЃ
gru_cell_55/BiasAddBiasAddgru_cell_55/MatMul:product:0gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAddh
gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_55/Const
gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split/split_dimм
gru_cell_55/splitSplit$gru_cell_55/split/split_dim:output:0gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/splitЗ
#gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_55/MatMul_1/ReadVariableOpЅ
gru_cell_55/MatMul_1MatMulzeros:output:0+gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMul_1Љ
gru_cell_55/BiasAdd_1BiasAddgru_cell_55/MatMul_1:product:0gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAdd_1
gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_55/Const_1
gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split_1/split_dim
gru_cell_55/split_1SplitVgru_cell_55/BiasAdd_1:output:0gru_cell_55/Const_1:output:0&gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/split_1
gru_cell_55/addAddV2gru_cell_55/split:output:0gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add|
gru_cell_55/SigmoidSigmoidgru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid
gru_cell_55/add_1AddV2gru_cell_55/split:output:1gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_1
gru_cell_55/Sigmoid_1Sigmoidgru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid_1
gru_cell_55/mulMulgru_cell_55/Sigmoid_1:y:0gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul
gru_cell_55/add_2AddV2gru_cell_55/split:output:2gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_2u
gru_cell_55/TanhTanhgru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Tanh
gru_cell_55/mul_1Mulgru_cell_55/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_1k
gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_55/sub/x
gru_cell_55/subSubgru_cell_55/sub/x:output:0gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/sub
gru_cell_55/mul_2Mulgru_cell_55/sub:z:0gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_2
gru_cell_55/add_3AddV2gru_cell_55/mul_1:z:0gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_55_readvariableop_resource*gru_cell_55_matmul_readvariableop_resource,gru_cell_55_matmul_1_readvariableop_resource*
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
while_body_1126675*
condR
while_cond_1126674*8
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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1126418

inputs'
#gru_cell_54_readvariableop_resource.
*gru_cell_54_matmul_readvariableop_resource0
,gru_cell_54_matmul_1_readvariableop_resource
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
gru_cell_54/ReadVariableOpReadVariableOp#gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_54/ReadVariableOp
gru_cell_54/unstackUnpack"gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_54/unstackБ
!gru_cell_54/MatMul/ReadVariableOpReadVariableOp*gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_54/MatMul/ReadVariableOpЉ
gru_cell_54/MatMulMatMulstrided_slice_2:output:0)gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMulЃ
gru_cell_54/BiasAddBiasAddgru_cell_54/MatMul:product:0gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAddh
gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_54/Const
gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split/split_dimм
gru_cell_54/splitSplit$gru_cell_54/split/split_dim:output:0gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/splitЗ
#gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_54/MatMul_1/ReadVariableOpЅ
gru_cell_54/MatMul_1MatMulzeros:output:0+gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMul_1Љ
gru_cell_54/BiasAdd_1BiasAddgru_cell_54/MatMul_1:product:0gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAdd_1
gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_54/Const_1
gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split_1/split_dim
gru_cell_54/split_1SplitVgru_cell_54/BiasAdd_1:output:0gru_cell_54/Const_1:output:0&gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/split_1
gru_cell_54/addAddV2gru_cell_54/split:output:0gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add|
gru_cell_54/SigmoidSigmoidgru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid
gru_cell_54/add_1AddV2gru_cell_54/split:output:1gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_1
gru_cell_54/Sigmoid_1Sigmoidgru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid_1
gru_cell_54/mulMulgru_cell_54/Sigmoid_1:y:0gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul
gru_cell_54/add_2AddV2gru_cell_54/split:output:2gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_2u
gru_cell_54/TanhTanhgru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Tanh
gru_cell_54/mul_1Mulgru_cell_54/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_1k
gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_54/sub/x
gru_cell_54/subSubgru_cell_54/sub/x:output:0gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/sub
gru_cell_54/mul_2Mulgru_cell_54/sub:z:0gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_2
gru_cell_54/add_3AddV2gru_cell_54/mul_1:z:0gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_54_readvariableop_resource*gru_cell_54_matmul_readvariableop_resource,gru_cell_54_matmul_1_readvariableop_resource*
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
while_body_1126328*
condR
while_cond_1126327*8
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
џ!
т
while_body_1125906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_55_1125928_0
while_gru_cell_55_1125930_0
while_gru_cell_55_1125932_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_55_1125928
while_gru_cell_55_1125930
while_gru_cell_55_1125932Ђ)while/gru_cell_55/StatefulPartitionedCallУ
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
)while/gru_cell_55/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_55_1125928_0while_gru_cell_55_1125930_0while_gru_cell_55_1125932_0*
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_11256072+
)while/gru_cell_55/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_55/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_55/StatefulPartitionedCall:output:1*^while/gru_cell_55/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_55_1125928while_gru_cell_55_1125928_0"8
while_gru_cell_55_1125930while_gru_cell_55_1125930_0"8
while_gru_cell_55_1125932while_gru_cell_55_1125932_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_55/StatefulPartitionedCall)while/gru_cell_55/StatefulPartitionedCall: 
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129764

inputs'
#gru_cell_55_readvariableop_resource.
*gru_cell_55_matmul_readvariableop_resource0
,gru_cell_55_matmul_1_readvariableop_resource
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
gru_cell_55/ReadVariableOpReadVariableOp#gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_55/ReadVariableOp
gru_cell_55/unstackUnpack"gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_55/unstackБ
!gru_cell_55/MatMul/ReadVariableOpReadVariableOp*gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_55/MatMul/ReadVariableOpЉ
gru_cell_55/MatMulMatMulstrided_slice_2:output:0)gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMulЃ
gru_cell_55/BiasAddBiasAddgru_cell_55/MatMul:product:0gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAddh
gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_55/Const
gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split/split_dimм
gru_cell_55/splitSplit$gru_cell_55/split/split_dim:output:0gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/splitЗ
#gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_55/MatMul_1/ReadVariableOpЅ
gru_cell_55/MatMul_1MatMulzeros:output:0+gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMul_1Љ
gru_cell_55/BiasAdd_1BiasAddgru_cell_55/MatMul_1:product:0gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAdd_1
gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_55/Const_1
gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split_1/split_dim
gru_cell_55/split_1SplitVgru_cell_55/BiasAdd_1:output:0gru_cell_55/Const_1:output:0&gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/split_1
gru_cell_55/addAddV2gru_cell_55/split:output:0gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add|
gru_cell_55/SigmoidSigmoidgru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid
gru_cell_55/add_1AddV2gru_cell_55/split:output:1gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_1
gru_cell_55/Sigmoid_1Sigmoidgru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid_1
gru_cell_55/mulMulgru_cell_55/Sigmoid_1:y:0gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul
gru_cell_55/add_2AddV2gru_cell_55/split:output:2gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_2u
gru_cell_55/TanhTanhgru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Tanh
gru_cell_55/mul_1Mulgru_cell_55/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_1k
gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_55/sub/x
gru_cell_55/subSubgru_cell_55/sub/x:output:0gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/sub
gru_cell_55/mul_2Mulgru_cell_55/sub:z:0gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_2
gru_cell_55/add_3AddV2gru_cell_55/mul_1:z:0gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_55_readvariableop_resource*gru_cell_55_matmul_readvariableop_resource,gru_cell_55_matmul_1_readvariableop_resource*
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
while_body_1129674*
condR
while_cond_1129673*8
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
е
Џ
while_cond_1125343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1125343___redundant_placeholder05
1while_while_cond_1125343___redundant_placeholder15
1while_while_cond_1125343___redundant_placeholder25
1while_while_cond_1125343___redundant_placeholder3
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
while_cond_1125905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1125905___redundant_placeholder05
1while_while_cond_1125905___redundant_placeholder15
1while_while_cond_1125905___redundant_placeholder25
1while_while_cond_1125905___redundant_placeholder3
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
while_body_1126169
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_54_readvariableop_resource_06
2while_gru_cell_54_matmul_readvariableop_resource_08
4while_gru_cell_54_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_54_readvariableop_resource4
0while_gru_cell_54_matmul_readvariableop_resource6
2while_gru_cell_54_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_54/ReadVariableOpReadVariableOp+while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_54/ReadVariableOp 
while/gru_cell_54/unstackUnpack(while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_54/unstackХ
'while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_54/MatMul/ReadVariableOpг
while/gru_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMulЛ
while/gru_cell_54/BiasAddBiasAdd"while/gru_cell_54/MatMul:product:0"while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAddt
while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_54/Const
!while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_54/split/split_dimє
while/gru_cell_54/splitSplit*while/gru_cell_54/split/split_dim:output:0"while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/splitЫ
)while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_54/MatMul_1/ReadVariableOpМ
while/gru_cell_54/MatMul_1MatMulwhile_placeholder_21while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMul_1С
while/gru_cell_54/BiasAdd_1BiasAdd$while/gru_cell_54/MatMul_1:product:0"while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAdd_1
while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_54/Const_1
#while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_54/split_1/split_dim­
while/gru_cell_54/split_1SplitV$while/gru_cell_54/BiasAdd_1:output:0"while/gru_cell_54/Const_1:output:0,while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/split_1Џ
while/gru_cell_54/addAddV2 while/gru_cell_54/split:output:0"while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add
while/gru_cell_54/SigmoidSigmoidwhile/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/SigmoidГ
while/gru_cell_54/add_1AddV2 while/gru_cell_54/split:output:1"while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_1
while/gru_cell_54/Sigmoid_1Sigmoidwhile/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Sigmoid_1Ќ
while/gru_cell_54/mulMulwhile/gru_cell_54/Sigmoid_1:y:0"while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mulЊ
while/gru_cell_54/add_2AddV2 while/gru_cell_54/split:output:2while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_2
while/gru_cell_54/TanhTanhwhile/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Tanh
while/gru_cell_54/mul_1Mulwhile/gru_cell_54/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_1w
while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_54/sub/xЈ
while/gru_cell_54/subSub while/gru_cell_54/sub/x:output:0while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/subЂ
while/gru_cell_54/mul_2Mulwhile/gru_cell_54/sub:z:0while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_2Ї
while/gru_cell_54/add_3AddV2while/gru_cell_54/mul_1:z:0while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_54/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_54_matmul_1_readvariableop_resource4while_gru_cell_54_matmul_1_readvariableop_resource_0"f
0while_gru_cell_54_matmul_readvariableop_resource2while_gru_cell_54_matmul_readvariableop_resource_0"X
)while_gru_cell_54_readvariableop_resource+while_gru_cell_54_readvariableop_resource_0")
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
і<
к
B__inference_GRU_1_layer_call_and_return_conditional_losses_1125408

inputs
gru_cell_54_1125332
gru_cell_54_1125334
gru_cell_54_1125336
identityЂ#gru_cell_54/StatefulPartitionedCallЂwhileD
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
#gru_cell_54/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_54_1125332gru_cell_54_1125334gru_cell_54_1125336*
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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_11250452%
#gru_cell_54/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_54_1125332gru_cell_54_1125334gru_cell_54_1125336*
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
while_body_1125344*
condR
while_cond_1125343*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_54/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_54/StatefulPartitionedCall#gru_cell_54/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г

'__inference_GRU_1_layer_call_fn_1129095
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_11254082
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


'__inference_GRU_2_layer_call_fn_1129775

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
B__inference_GRU_2_layer_call_and_return_conditional_losses_11266062
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
е
Џ
while_cond_1128834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1128834___redundant_placeholder05
1while_while_cond_1128834___redundant_placeholder15
1while_while_cond_1128834___redundant_placeholder25
1while_while_cond_1128834___redundant_placeholder3
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
-__inference_gru_cell_54_layer_call_fn_1129934

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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_11250852
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
while_cond_1128993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1128993___redundant_placeholder05
1while_while_cond_1128993___redundant_placeholder15
1while_while_cond_1128993___redundant_placeholder25
1while_while_cond_1128993___redundant_placeholder3
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
GRU_2_while_cond_1127201(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1127201___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1127201___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1127201___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1127201___redundant_placeholder3
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
оH
и
GRU_2_while_body_1128267(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_55_readvariableop_resource_0<
8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_55_readvariableop_resource:
6gru_2_while_gru_cell_55_matmul_readvariableop_resource<
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_55/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_55/ReadVariableOpВ
GRU_2/while/gru_cell_55/unstackUnpack.GRU_2/while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_55/unstackз
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_55/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_55/MatMulг
GRU_2/while/gru_cell_55/BiasAddBiasAdd(GRU_2/while/gru_cell_55/MatMul:product:0(GRU_2/while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_55/BiasAdd
GRU_2/while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_55/Const
'GRU_2/while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_55/split/split_dim
GRU_2/while/gru_cell_55/splitSplit0GRU_2/while/gru_cell_55/split/split_dim:output:0(GRU_2/while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_55/splitн
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_55/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_55/MatMul_1й
!GRU_2/while/gru_cell_55/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_55/MatMul_1:product:0(GRU_2/while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_55/BiasAdd_1
GRU_2/while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_55/Const_1Ё
)GRU_2/while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_55/split_1/split_dimЫ
GRU_2/while/gru_cell_55/split_1SplitV*GRU_2/while/gru_cell_55/BiasAdd_1:output:0(GRU_2/while/gru_cell_55/Const_1:output:02GRU_2/while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_55/split_1Ч
GRU_2/while/gru_cell_55/addAddV2&GRU_2/while/gru_cell_55/split:output:0(GRU_2/while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add 
GRU_2/while/gru_cell_55/SigmoidSigmoidGRU_2/while/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_55/SigmoidЫ
GRU_2/while/gru_cell_55/add_1AddV2&GRU_2/while/gru_cell_55/split:output:1(GRU_2/while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_1І
!GRU_2/while/gru_cell_55/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_55/Sigmoid_1Ф
GRU_2/while/gru_cell_55/mulMul%GRU_2/while/gru_cell_55/Sigmoid_1:y:0(GRU_2/while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mulТ
GRU_2/while/gru_cell_55/add_2AddV2&GRU_2/while/gru_cell_55/split:output:2GRU_2/while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_2
GRU_2/while/gru_cell_55/TanhTanh!GRU_2/while/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/TanhЗ
GRU_2/while/gru_cell_55/mul_1Mul#GRU_2/while/gru_cell_55/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_1
GRU_2/while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_55/sub/xР
GRU_2/while/gru_cell_55/subSub&GRU_2/while/gru_cell_55/sub/x:output:0#GRU_2/while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/subК
GRU_2/while/gru_cell_55/mul_2MulGRU_2/while/gru_cell_55/sub:z:0 GRU_2/while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_2П
GRU_2/while/gru_cell_55/add_3AddV2!GRU_2/while/gru_cell_55/mul_1:z:0!GRU_2/while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_55/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resource:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_55_matmul_readvariableop_resource8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_55_readvariableop_resource1gru_2_while_gru_cell_55_readvariableop_resource_0"5
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
Й
и
G__inference_Supervisor_layer_call_and_return_conditional_losses_1126892

inputs
gru_1_1126872
gru_1_1126874
gru_1_1126876
gru_2_1126879
gru_2_1126881
gru_2_1126883
out_1126886
out_1126888
identityЂGRU_1/StatefulPartitionedCallЂGRU_2/StatefulPartitionedCallЂOUT/StatefulPartitionedCall
GRU_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_1126872gru_1_1126874gru_1_1126876*
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_11262592
GRU_1/StatefulPartitionedCallН
GRU_2/StatefulPartitionedCallStatefulPartitionedCall&GRU_1/StatefulPartitionedCall:output:0gru_2_1126879gru_2_1126881gru_2_1126883*
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_11266062
GRU_2/StatefulPartitionedCallЂ
OUT/StatefulPartitionedCallStatefulPartitionedCall&GRU_2/StatefulPartitionedCall:output:0out_1126886out_1126888*
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
@__inference_OUT_layer_call_and_return_conditional_losses_11268262
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

й
%__inference_signature_wrapper_1126978
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
"__inference__wrapped_model_11249732
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
ыW
є
B__inference_GRU_2_layer_call_and_return_conditional_losses_1126606

inputs'
#gru_cell_55_readvariableop_resource.
*gru_cell_55_matmul_readvariableop_resource0
,gru_cell_55_matmul_1_readvariableop_resource
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
gru_cell_55/ReadVariableOpReadVariableOp#gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_55/ReadVariableOp
gru_cell_55/unstackUnpack"gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_55/unstackБ
!gru_cell_55/MatMul/ReadVariableOpReadVariableOp*gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_55/MatMul/ReadVariableOpЉ
gru_cell_55/MatMulMatMulstrided_slice_2:output:0)gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMulЃ
gru_cell_55/BiasAddBiasAddgru_cell_55/MatMul:product:0gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAddh
gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_55/Const
gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split/split_dimм
gru_cell_55/splitSplit$gru_cell_55/split/split_dim:output:0gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/splitЗ
#gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_55/MatMul_1/ReadVariableOpЅ
gru_cell_55/MatMul_1MatMulzeros:output:0+gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMul_1Љ
gru_cell_55/BiasAdd_1BiasAddgru_cell_55/MatMul_1:product:0gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAdd_1
gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_55/Const_1
gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split_1/split_dim
gru_cell_55/split_1SplitVgru_cell_55/BiasAdd_1:output:0gru_cell_55/Const_1:output:0&gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/split_1
gru_cell_55/addAddV2gru_cell_55/split:output:0gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add|
gru_cell_55/SigmoidSigmoidgru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid
gru_cell_55/add_1AddV2gru_cell_55/split:output:1gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_1
gru_cell_55/Sigmoid_1Sigmoidgru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid_1
gru_cell_55/mulMulgru_cell_55/Sigmoid_1:y:0gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul
gru_cell_55/add_2AddV2gru_cell_55/split:output:2gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_2u
gru_cell_55/TanhTanhgru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Tanh
gru_cell_55/mul_1Mulgru_cell_55/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_1k
gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_55/sub/x
gru_cell_55/subSubgru_cell_55/sub/x:output:0gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/sub
gru_cell_55/mul_2Mulgru_cell_55/sub:z:0gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_2
gru_cell_55/add_3AddV2gru_cell_55/mul_1:z:0gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_55_readvariableop_resource*gru_cell_55_matmul_readvariableop_resource,gru_cell_55_matmul_1_readvariableop_resource*
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
while_body_1126516*
condR
while_cond_1126515*8
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
е
Џ
while_cond_1128653
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1128653___redundant_placeholder05
1while_while_cond_1128653___redundant_placeholder15
1while_while_cond_1128653___redundant_placeholder25
1while_while_cond_1128653___redundant_placeholder3
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
GRU_2_while_cond_1128266(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1128266___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1128266___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1128266___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1128266___redundant_placeholder3
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
	
Ё
GRU_1_while_cond_1127387(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1127387___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1127387___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1127387___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1127387___redundant_placeholder3
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
ыW
є
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128744

inputs'
#gru_cell_54_readvariableop_resource.
*gru_cell_54_matmul_readvariableop_resource0
,gru_cell_54_matmul_1_readvariableop_resource
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
gru_cell_54/ReadVariableOpReadVariableOp#gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_54/ReadVariableOp
gru_cell_54/unstackUnpack"gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_54/unstackБ
!gru_cell_54/MatMul/ReadVariableOpReadVariableOp*gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_54/MatMul/ReadVariableOpЉ
gru_cell_54/MatMulMatMulstrided_slice_2:output:0)gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMulЃ
gru_cell_54/BiasAddBiasAddgru_cell_54/MatMul:product:0gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAddh
gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_54/Const
gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split/split_dimм
gru_cell_54/splitSplit$gru_cell_54/split/split_dim:output:0gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/splitЗ
#gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_54/MatMul_1/ReadVariableOpЅ
gru_cell_54/MatMul_1MatMulzeros:output:0+gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMul_1Љ
gru_cell_54/BiasAdd_1BiasAddgru_cell_54/MatMul_1:product:0gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAdd_1
gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_54/Const_1
gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split_1/split_dim
gru_cell_54/split_1SplitVgru_cell_54/BiasAdd_1:output:0gru_cell_54/Const_1:output:0&gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/split_1
gru_cell_54/addAddV2gru_cell_54/split:output:0gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add|
gru_cell_54/SigmoidSigmoidgru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid
gru_cell_54/add_1AddV2gru_cell_54/split:output:1gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_1
gru_cell_54/Sigmoid_1Sigmoidgru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid_1
gru_cell_54/mulMulgru_cell_54/Sigmoid_1:y:0gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul
gru_cell_54/add_2AddV2gru_cell_54/split:output:2gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_2u
gru_cell_54/TanhTanhgru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Tanh
gru_cell_54/mul_1Mulgru_cell_54/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_1k
gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_54/sub/x
gru_cell_54/subSubgru_cell_54/sub/x:output:0gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/sub
gru_cell_54/mul_2Mulgru_cell_54/sub:z:0gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_2
gru_cell_54/add_3AddV2gru_cell_54/mul_1:z:0gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_54_readvariableop_resource*gru_cell_54_matmul_readvariableop_resource,gru_cell_54_matmul_1_readvariableop_resource*
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
while_body_1128654*
condR
while_cond_1128653*8
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
Ф
ђ
#Supervisor_GRU_1_while_cond_1124700>
:supervisor_gru_1_while_supervisor_gru_1_while_loop_counterD
@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations&
"supervisor_gru_1_while_placeholder(
$supervisor_gru_1_while_placeholder_1(
$supervisor_gru_1_while_placeholder_2@
<supervisor_gru_1_while_less_supervisor_gru_1_strided_slice_1W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1124700___redundant_placeholder0W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1124700___redundant_placeholder1W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1124700___redundant_placeholder2W
Ssupervisor_gru_1_while_supervisor_gru_1_while_cond_1124700___redundant_placeholder3#
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
@
Ж
while_body_1129175
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_55_readvariableop_resource_06
2while_gru_cell_55_matmul_readvariableop_resource_08
4while_gru_cell_55_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_55_readvariableop_resource4
0while_gru_cell_55_matmul_readvariableop_resource6
2while_gru_cell_55_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_55/ReadVariableOpReadVariableOp+while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_55/ReadVariableOp 
while/gru_cell_55/unstackUnpack(while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_55/unstackХ
'while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_55/MatMul/ReadVariableOpг
while/gru_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMulЛ
while/gru_cell_55/BiasAddBiasAdd"while/gru_cell_55/MatMul:product:0"while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAddt
while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_55/Const
!while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_55/split/split_dimє
while/gru_cell_55/splitSplit*while/gru_cell_55/split/split_dim:output:0"while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/splitЫ
)while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_55/MatMul_1/ReadVariableOpМ
while/gru_cell_55/MatMul_1MatMulwhile_placeholder_21while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMul_1С
while/gru_cell_55/BiasAdd_1BiasAdd$while/gru_cell_55/MatMul_1:product:0"while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAdd_1
while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_55/Const_1
#while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_55/split_1/split_dim­
while/gru_cell_55/split_1SplitV$while/gru_cell_55/BiasAdd_1:output:0"while/gru_cell_55/Const_1:output:0,while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/split_1Џ
while/gru_cell_55/addAddV2 while/gru_cell_55/split:output:0"while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add
while/gru_cell_55/SigmoidSigmoidwhile/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/SigmoidГ
while/gru_cell_55/add_1AddV2 while/gru_cell_55/split:output:1"while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_1
while/gru_cell_55/Sigmoid_1Sigmoidwhile/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Sigmoid_1Ќ
while/gru_cell_55/mulMulwhile/gru_cell_55/Sigmoid_1:y:0"while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mulЊ
while/gru_cell_55/add_2AddV2 while/gru_cell_55/split:output:2while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_2
while/gru_cell_55/TanhTanhwhile/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Tanh
while/gru_cell_55/mul_1Mulwhile/gru_cell_55/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_1w
while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_55/sub/xЈ
while/gru_cell_55/subSub while/gru_cell_55/sub/x:output:0while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/subЂ
while/gru_cell_55/mul_2Mulwhile/gru_cell_55/sub:z:0while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_2Ї
while/gru_cell_55/add_3AddV2while/gru_cell_55/mul_1:z:0while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_55/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_55_matmul_1_readvariableop_resource4while_gru_cell_55_matmul_1_readvariableop_resource_0"f
0while_gru_cell_55_matmul_readvariableop_resource2while_gru_cell_55_matmul_readvariableop_resource_0"X
)while_gru_cell_55_readvariableop_resource+while_gru_cell_55_readvariableop_resource_0")
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
ус

G__inference_Supervisor_layer_call_and_return_conditional_losses_1127660
gru_1_input-
)gru_1_gru_cell_54_readvariableop_resource4
0gru_1_gru_cell_54_matmul_readvariableop_resource6
2gru_1_gru_cell_54_matmul_1_readvariableop_resource-
)gru_2_gru_cell_55_readvariableop_resource4
0gru_2_gru_cell_55_matmul_readvariableop_resource6
2gru_2_gru_cell_55_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_54/ReadVariableOpReadVariableOp)gru_1_gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_54/ReadVariableOp 
GRU_1/gru_cell_54/unstackUnpack(GRU_1/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_54/unstackУ
'GRU_1/gru_cell_54/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_54/MatMul/ReadVariableOpС
GRU_1/gru_cell_54/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMulЛ
GRU_1/gru_cell_54/BiasAddBiasAdd"GRU_1/gru_cell_54/MatMul:product:0"GRU_1/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAddt
GRU_1/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_54/Const
!GRU_1/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_54/split/split_dimє
GRU_1/gru_cell_54/splitSplit*GRU_1/gru_cell_54/split/split_dim:output:0"GRU_1/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/splitЩ
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_54/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMul_1С
GRU_1/gru_cell_54/BiasAdd_1BiasAdd$GRU_1/gru_cell_54/MatMul_1:product:0"GRU_1/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAdd_1
GRU_1/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_54/Const_1
#GRU_1/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_54/split_1/split_dim­
GRU_1/gru_cell_54/split_1SplitV$GRU_1/gru_cell_54/BiasAdd_1:output:0"GRU_1/gru_cell_54/Const_1:output:0,GRU_1/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/split_1Џ
GRU_1/gru_cell_54/addAddV2 GRU_1/gru_cell_54/split:output:0"GRU_1/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add
GRU_1/gru_cell_54/SigmoidSigmoidGRU_1/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/SigmoidГ
GRU_1/gru_cell_54/add_1AddV2 GRU_1/gru_cell_54/split:output:1"GRU_1/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_1
GRU_1/gru_cell_54/Sigmoid_1SigmoidGRU_1/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Sigmoid_1Ќ
GRU_1/gru_cell_54/mulMulGRU_1/gru_cell_54/Sigmoid_1:y:0"GRU_1/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mulЊ
GRU_1/gru_cell_54/add_2AddV2 GRU_1/gru_cell_54/split:output:2GRU_1/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_2
GRU_1/gru_cell_54/TanhTanhGRU_1/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Tanh 
GRU_1/gru_cell_54/mul_1MulGRU_1/gru_cell_54/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_1w
GRU_1/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_54/sub/xЈ
GRU_1/gru_cell_54/subSub GRU_1/gru_cell_54/sub/x:output:0GRU_1/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/subЂ
GRU_1/gru_cell_54/mul_2MulGRU_1/gru_cell_54/sub:z:0GRU_1/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_2Ї
GRU_1/gru_cell_54/add_3AddV2GRU_1/gru_cell_54/mul_1:z:0GRU_1/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_3
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
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_54_readvariableop_resource0gru_1_gru_cell_54_matmul_readvariableop_resource2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
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
GRU_1_while_body_1127388*$
condR
GRU_1_while_cond_1127387*8
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
 GRU_2/gru_cell_55/ReadVariableOpReadVariableOp)gru_2_gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_55/ReadVariableOp 
GRU_2/gru_cell_55/unstackUnpack(GRU_2/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_55/unstackУ
'GRU_2/gru_cell_55/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_55/MatMul/ReadVariableOpС
GRU_2/gru_cell_55/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMulЛ
GRU_2/gru_cell_55/BiasAddBiasAdd"GRU_2/gru_cell_55/MatMul:product:0"GRU_2/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAddt
GRU_2/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_55/Const
!GRU_2/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_55/split/split_dimє
GRU_2/gru_cell_55/splitSplit*GRU_2/gru_cell_55/split/split_dim:output:0"GRU_2/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/splitЩ
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_55/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMul_1С
GRU_2/gru_cell_55/BiasAdd_1BiasAdd$GRU_2/gru_cell_55/MatMul_1:product:0"GRU_2/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAdd_1
GRU_2/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_55/Const_1
#GRU_2/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_55/split_1/split_dim­
GRU_2/gru_cell_55/split_1SplitV$GRU_2/gru_cell_55/BiasAdd_1:output:0"GRU_2/gru_cell_55/Const_1:output:0,GRU_2/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/split_1Џ
GRU_2/gru_cell_55/addAddV2 GRU_2/gru_cell_55/split:output:0"GRU_2/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add
GRU_2/gru_cell_55/SigmoidSigmoidGRU_2/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/SigmoidГ
GRU_2/gru_cell_55/add_1AddV2 GRU_2/gru_cell_55/split:output:1"GRU_2/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_1
GRU_2/gru_cell_55/Sigmoid_1SigmoidGRU_2/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Sigmoid_1Ќ
GRU_2/gru_cell_55/mulMulGRU_2/gru_cell_55/Sigmoid_1:y:0"GRU_2/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mulЊ
GRU_2/gru_cell_55/add_2AddV2 GRU_2/gru_cell_55/split:output:2GRU_2/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_2
GRU_2/gru_cell_55/TanhTanhGRU_2/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Tanh 
GRU_2/gru_cell_55/mul_1MulGRU_2/gru_cell_55/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_1w
GRU_2/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_55/sub/xЈ
GRU_2/gru_cell_55/subSub GRU_2/gru_cell_55/sub/x:output:0GRU_2/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/subЂ
GRU_2/gru_cell_55/mul_2MulGRU_2/gru_cell_55/sub:z:0GRU_2/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_2Ї
GRU_2/gru_cell_55/add_3AddV2GRU_2/gru_cell_55/mul_1:z:0GRU_2/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_3
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
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_55_readvariableop_resource0gru_2_gru_cell_55_matmul_readvariableop_resource2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
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
GRU_2_while_body_1127543*$
condR
GRU_2_while_cond_1127542*8
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
Яс

G__inference_Supervisor_layer_call_and_return_conditional_losses_1128384

inputs-
)gru_1_gru_cell_54_readvariableop_resource4
0gru_1_gru_cell_54_matmul_readvariableop_resource6
2gru_1_gru_cell_54_matmul_1_readvariableop_resource-
)gru_2_gru_cell_55_readvariableop_resource4
0gru_2_gru_cell_55_matmul_readvariableop_resource6
2gru_2_gru_cell_55_matmul_1_readvariableop_resource)
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
 GRU_1/gru_cell_54/ReadVariableOpReadVariableOp)gru_1_gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_1/gru_cell_54/ReadVariableOp 
GRU_1/gru_cell_54/unstackUnpack(GRU_1/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_1/gru_cell_54/unstackУ
'GRU_1/gru_cell_54/MatMul/ReadVariableOpReadVariableOp0gru_1_gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_1/gru_cell_54/MatMul/ReadVariableOpС
GRU_1/gru_cell_54/MatMulMatMulGRU_1/strided_slice_2:output:0/GRU_1/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMulЛ
GRU_1/gru_cell_54/BiasAddBiasAdd"GRU_1/gru_cell_54/MatMul:product:0"GRU_1/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAddt
GRU_1/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/gru_cell_54/Const
!GRU_1/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_1/gru_cell_54/split/split_dimє
GRU_1/gru_cell_54/splitSplit*GRU_1/gru_cell_54/split/split_dim:output:0"GRU_1/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/splitЩ
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_1/gru_cell_54/MatMul_1/ReadVariableOpН
GRU_1/gru_cell_54/MatMul_1MatMulGRU_1/zeros:output:01GRU_1/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/MatMul_1С
GRU_1/gru_cell_54/BiasAdd_1BiasAdd$GRU_1/gru_cell_54/MatMul_1:product:0"GRU_1/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_1/gru_cell_54/BiasAdd_1
GRU_1/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_1/gru_cell_54/Const_1
#GRU_1/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_1/gru_cell_54/split_1/split_dim­
GRU_1/gru_cell_54/split_1SplitV$GRU_1/gru_cell_54/BiasAdd_1:output:0"GRU_1/gru_cell_54/Const_1:output:0,GRU_1/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/gru_cell_54/split_1Џ
GRU_1/gru_cell_54/addAddV2 GRU_1/gru_cell_54/split:output:0"GRU_1/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add
GRU_1/gru_cell_54/SigmoidSigmoidGRU_1/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/SigmoidГ
GRU_1/gru_cell_54/add_1AddV2 GRU_1/gru_cell_54/split:output:1"GRU_1/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_1
GRU_1/gru_cell_54/Sigmoid_1SigmoidGRU_1/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Sigmoid_1Ќ
GRU_1/gru_cell_54/mulMulGRU_1/gru_cell_54/Sigmoid_1:y:0"GRU_1/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mulЊ
GRU_1/gru_cell_54/add_2AddV2 GRU_1/gru_cell_54/split:output:2GRU_1/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_2
GRU_1/gru_cell_54/TanhTanhGRU_1/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/Tanh 
GRU_1/gru_cell_54/mul_1MulGRU_1/gru_cell_54/Sigmoid:y:0GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_1w
GRU_1/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/gru_cell_54/sub/xЈ
GRU_1/gru_cell_54/subSub GRU_1/gru_cell_54/sub/x:output:0GRU_1/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/subЂ
GRU_1/gru_cell_54/mul_2MulGRU_1/gru_cell_54/sub:z:0GRU_1/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/mul_2Ї
GRU_1/gru_cell_54/add_3AddV2GRU_1/gru_cell_54/mul_1:z:0GRU_1/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/gru_cell_54/add_3
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
GRU_1/whileWhile!GRU_1/while/loop_counter:output:0'GRU_1/while/maximum_iterations:output:0GRU_1/time:output:0GRU_1/TensorArrayV2_1:handle:0GRU_1/zeros:output:0GRU_1/strided_slice_1:output:0=GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_1_gru_cell_54_readvariableop_resource0gru_1_gru_cell_54_matmul_readvariableop_resource2gru_1_gru_cell_54_matmul_1_readvariableop_resource*
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
GRU_1_while_body_1128112*$
condR
GRU_1_while_cond_1128111*8
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
 GRU_2/gru_cell_55/ReadVariableOpReadVariableOp)gru_2_gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02"
 GRU_2/gru_cell_55/ReadVariableOp 
GRU_2/gru_cell_55/unstackUnpack(GRU_2/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
GRU_2/gru_cell_55/unstackУ
'GRU_2/gru_cell_55/MatMul/ReadVariableOpReadVariableOp0gru_2_gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'GRU_2/gru_cell_55/MatMul/ReadVariableOpС
GRU_2/gru_cell_55/MatMulMatMulGRU_2/strided_slice_2:output:0/GRU_2/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMulЛ
GRU_2/gru_cell_55/BiasAddBiasAdd"GRU_2/gru_cell_55/MatMul:product:0"GRU_2/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAddt
GRU_2/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/gru_cell_55/Const
!GRU_2/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!GRU_2/gru_cell_55/split/split_dimє
GRU_2/gru_cell_55/splitSplit*GRU_2/gru_cell_55/split/split_dim:output:0"GRU_2/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/splitЩ
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02+
)GRU_2/gru_cell_55/MatMul_1/ReadVariableOpН
GRU_2/gru_cell_55/MatMul_1MatMulGRU_2/zeros:output:01GRU_2/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/MatMul_1С
GRU_2/gru_cell_55/BiasAdd_1BiasAdd$GRU_2/gru_cell_55/MatMul_1:product:0"GRU_2/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
GRU_2/gru_cell_55/BiasAdd_1
GRU_2/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
GRU_2/gru_cell_55/Const_1
#GRU_2/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#GRU_2/gru_cell_55/split_1/split_dim­
GRU_2/gru_cell_55/split_1SplitV$GRU_2/gru_cell_55/BiasAdd_1:output:0"GRU_2/gru_cell_55/Const_1:output:0,GRU_2/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/gru_cell_55/split_1Џ
GRU_2/gru_cell_55/addAddV2 GRU_2/gru_cell_55/split:output:0"GRU_2/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add
GRU_2/gru_cell_55/SigmoidSigmoidGRU_2/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/SigmoidГ
GRU_2/gru_cell_55/add_1AddV2 GRU_2/gru_cell_55/split:output:1"GRU_2/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_1
GRU_2/gru_cell_55/Sigmoid_1SigmoidGRU_2/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Sigmoid_1Ќ
GRU_2/gru_cell_55/mulMulGRU_2/gru_cell_55/Sigmoid_1:y:0"GRU_2/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mulЊ
GRU_2/gru_cell_55/add_2AddV2 GRU_2/gru_cell_55/split:output:2GRU_2/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_2
GRU_2/gru_cell_55/TanhTanhGRU_2/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/Tanh 
GRU_2/gru_cell_55/mul_1MulGRU_2/gru_cell_55/Sigmoid:y:0GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_1w
GRU_2/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/gru_cell_55/sub/xЈ
GRU_2/gru_cell_55/subSub GRU_2/gru_cell_55/sub/x:output:0GRU_2/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/subЂ
GRU_2/gru_cell_55/mul_2MulGRU_2/gru_cell_55/sub:z:0GRU_2/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/mul_2Ї
GRU_2/gru_cell_55/add_3AddV2GRU_2/gru_cell_55/mul_1:z:0GRU_2/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/gru_cell_55/add_3
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
GRU_2/whileWhile!GRU_2/while/loop_counter:output:0'GRU_2/while/maximum_iterations:output:0GRU_2/time:output:0GRU_2/TensorArrayV2_1:handle:0GRU_2/zeros:output:0GRU_2/strided_slice_1:output:0=GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_2_gru_cell_55_readvariableop_resource0gru_2_gru_cell_55_matmul_readvariableop_resource2gru_2_gru_cell_55_matmul_1_readvariableop_resource*
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
GRU_2_while_body_1128267*$
condR
GRU_2_while_cond_1128266*8
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
@
Ж
while_body_1126675
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_55_readvariableop_resource_06
2while_gru_cell_55_matmul_readvariableop_resource_08
4while_gru_cell_55_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_55_readvariableop_resource4
0while_gru_cell_55_matmul_readvariableop_resource6
2while_gru_cell_55_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_55/ReadVariableOpReadVariableOp+while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_55/ReadVariableOp 
while/gru_cell_55/unstackUnpack(while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_55/unstackХ
'while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_55/MatMul/ReadVariableOpг
while/gru_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMulЛ
while/gru_cell_55/BiasAddBiasAdd"while/gru_cell_55/MatMul:product:0"while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAddt
while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_55/Const
!while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_55/split/split_dimє
while/gru_cell_55/splitSplit*while/gru_cell_55/split/split_dim:output:0"while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/splitЫ
)while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_55/MatMul_1/ReadVariableOpМ
while/gru_cell_55/MatMul_1MatMulwhile_placeholder_21while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMul_1С
while/gru_cell_55/BiasAdd_1BiasAdd$while/gru_cell_55/MatMul_1:product:0"while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAdd_1
while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_55/Const_1
#while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_55/split_1/split_dim­
while/gru_cell_55/split_1SplitV$while/gru_cell_55/BiasAdd_1:output:0"while/gru_cell_55/Const_1:output:0,while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/split_1Џ
while/gru_cell_55/addAddV2 while/gru_cell_55/split:output:0"while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add
while/gru_cell_55/SigmoidSigmoidwhile/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/SigmoidГ
while/gru_cell_55/add_1AddV2 while/gru_cell_55/split:output:1"while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_1
while/gru_cell_55/Sigmoid_1Sigmoidwhile/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Sigmoid_1Ќ
while/gru_cell_55/mulMulwhile/gru_cell_55/Sigmoid_1:y:0"while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mulЊ
while/gru_cell_55/add_2AddV2 while/gru_cell_55/split:output:2while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_2
while/gru_cell_55/TanhTanhwhile/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Tanh
while/gru_cell_55/mul_1Mulwhile/gru_cell_55/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_1w
while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_55/sub/xЈ
while/gru_cell_55/subSub while/gru_cell_55/sub/x:output:0while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/subЂ
while/gru_cell_55/mul_2Mulwhile/gru_cell_55/sub:z:0while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_2Ї
while/gru_cell_55/add_3AddV2while/gru_cell_55/mul_1:z:0while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_55/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_55_matmul_1_readvariableop_resource4while_gru_cell_55_matmul_1_readvariableop_resource_0"f
0while_gru_cell_55_matmul_readvariableop_resource2while_gru_cell_55_matmul_readvariableop_resource_0"X
)while_gru_cell_55_readvariableop_resource+while_gru_cell_55_readvariableop_resource_0")
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
	
Ё
GRU_1_while_cond_1127046(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1127046___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1127046___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1127046___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1127046___redundant_placeholder3
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
ч
ъ
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1125607

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
GRU_1_while_body_1127047(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_54_readvariableop_resource_0<
8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_54_readvariableop_resource:
6gru_1_while_gru_cell_54_matmul_readvariableop_resource<
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_54/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_54/ReadVariableOpВ
GRU_1/while/gru_cell_54/unstackUnpack.GRU_1/while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_54/unstackз
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_54/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_54/MatMulг
GRU_1/while/gru_cell_54/BiasAddBiasAdd(GRU_1/while/gru_cell_54/MatMul:product:0(GRU_1/while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_54/BiasAdd
GRU_1/while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_54/Const
'GRU_1/while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_54/split/split_dim
GRU_1/while/gru_cell_54/splitSplit0GRU_1/while/gru_cell_54/split/split_dim:output:0(GRU_1/while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_54/splitн
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_54/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_54/MatMul_1й
!GRU_1/while/gru_cell_54/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_54/MatMul_1:product:0(GRU_1/while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_54/BiasAdd_1
GRU_1/while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_54/Const_1Ё
)GRU_1/while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_54/split_1/split_dimЫ
GRU_1/while/gru_cell_54/split_1SplitV*GRU_1/while/gru_cell_54/BiasAdd_1:output:0(GRU_1/while/gru_cell_54/Const_1:output:02GRU_1/while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_54/split_1Ч
GRU_1/while/gru_cell_54/addAddV2&GRU_1/while/gru_cell_54/split:output:0(GRU_1/while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add 
GRU_1/while/gru_cell_54/SigmoidSigmoidGRU_1/while/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_54/SigmoidЫ
GRU_1/while/gru_cell_54/add_1AddV2&GRU_1/while/gru_cell_54/split:output:1(GRU_1/while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_1І
!GRU_1/while/gru_cell_54/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_54/Sigmoid_1Ф
GRU_1/while/gru_cell_54/mulMul%GRU_1/while/gru_cell_54/Sigmoid_1:y:0(GRU_1/while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mulТ
GRU_1/while/gru_cell_54/add_2AddV2&GRU_1/while/gru_cell_54/split:output:2GRU_1/while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_2
GRU_1/while/gru_cell_54/TanhTanh!GRU_1/while/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/TanhЗ
GRU_1/while/gru_cell_54/mul_1Mul#GRU_1/while/gru_cell_54/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_1
GRU_1/while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_54/sub/xР
GRU_1/while/gru_cell_54/subSub&GRU_1/while/gru_cell_54/sub/x:output:0#GRU_1/while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/subК
GRU_1/while/gru_cell_54/mul_2MulGRU_1/while/gru_cell_54/sub:z:0 GRU_1/while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_2П
GRU_1/while/gru_cell_54/add_3AddV2!GRU_1/while/gru_cell_54/mul_1:z:0!GRU_1/while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_54/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resource:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_54_matmul_readvariableop_resource8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_54_readvariableop_resource1gru_1_while_gru_cell_54_readvariableop_resource_0"5
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
Г
л
,__inference_Supervisor_layer_call_fn_1128426

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
G__inference_Supervisor_layer_call_and_return_conditional_losses_11269362
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
е
Џ
while_cond_1126168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1126168___redundant_placeholder05
1while_while_cond_1126168___redundant_placeholder15
1while_while_cond_1126168___redundant_placeholder25
1while_while_cond_1126168___redundant_placeholder3
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
-__inference_gru_cell_54_layer_call_fn_1129920

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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_11250452
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
@
Ж
while_body_1126328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_54_readvariableop_resource_06
2while_gru_cell_54_matmul_readvariableop_resource_08
4while_gru_cell_54_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_54_readvariableop_resource4
0while_gru_cell_54_matmul_readvariableop_resource6
2while_gru_cell_54_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_54/ReadVariableOpReadVariableOp+while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_54/ReadVariableOp 
while/gru_cell_54/unstackUnpack(while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_54/unstackХ
'while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_54/MatMul/ReadVariableOpг
while/gru_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMulЛ
while/gru_cell_54/BiasAddBiasAdd"while/gru_cell_54/MatMul:product:0"while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAddt
while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_54/Const
!while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_54/split/split_dimє
while/gru_cell_54/splitSplit*while/gru_cell_54/split/split_dim:output:0"while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/splitЫ
)while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_54/MatMul_1/ReadVariableOpМ
while/gru_cell_54/MatMul_1MatMulwhile_placeholder_21while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMul_1С
while/gru_cell_54/BiasAdd_1BiasAdd$while/gru_cell_54/MatMul_1:product:0"while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAdd_1
while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_54/Const_1
#while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_54/split_1/split_dim­
while/gru_cell_54/split_1SplitV$while/gru_cell_54/BiasAdd_1:output:0"while/gru_cell_54/Const_1:output:0,while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/split_1Џ
while/gru_cell_54/addAddV2 while/gru_cell_54/split:output:0"while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add
while/gru_cell_54/SigmoidSigmoidwhile/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/SigmoidГ
while/gru_cell_54/add_1AddV2 while/gru_cell_54/split:output:1"while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_1
while/gru_cell_54/Sigmoid_1Sigmoidwhile/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Sigmoid_1Ќ
while/gru_cell_54/mulMulwhile/gru_cell_54/Sigmoid_1:y:0"while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mulЊ
while/gru_cell_54/add_2AddV2 while/gru_cell_54/split:output:2while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_2
while/gru_cell_54/TanhTanhwhile/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Tanh
while/gru_cell_54/mul_1Mulwhile/gru_cell_54/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_1w
while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_54/sub/xЈ
while/gru_cell_54/subSub while/gru_cell_54/sub/x:output:0while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/subЂ
while/gru_cell_54/mul_2Mulwhile/gru_cell_54/sub:z:0while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_2Ї
while/gru_cell_54/add_3AddV2while/gru_cell_54/mul_1:z:0while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_54/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_54_matmul_1_readvariableop_resource4while_gru_cell_54_matmul_1_readvariableop_resource_0"f
0while_gru_cell_54_matmul_readvariableop_resource2while_gru_cell_54_matmul_readvariableop_resource_0"X
)while_gru_cell_54_readvariableop_resource+while_gru_cell_54_readvariableop_resource_0")
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
while_cond_1129333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1129333___redundant_placeholder05
1while_while_cond_1129333___redundant_placeholder15
1while_while_cond_1129333___redundant_placeholder25
1while_while_cond_1129333___redundant_placeholder3
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
ч
ъ
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1125085

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
while_cond_1126023
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1126023___redundant_placeholder05
1while_while_cond_1126023___redundant_placeholder15
1while_while_cond_1126023___redundant_placeholder25
1while_while_cond_1126023___redundant_placeholder3
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
while_body_1129334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_55_readvariableop_resource_06
2while_gru_cell_55_matmul_readvariableop_resource_08
4while_gru_cell_55_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_55_readvariableop_resource4
0while_gru_cell_55_matmul_readvariableop_resource6
2while_gru_cell_55_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_55/ReadVariableOpReadVariableOp+while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_55/ReadVariableOp 
while/gru_cell_55/unstackUnpack(while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_55/unstackХ
'while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_55/MatMul/ReadVariableOpг
while/gru_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMulЛ
while/gru_cell_55/BiasAddBiasAdd"while/gru_cell_55/MatMul:product:0"while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAddt
while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_55/Const
!while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_55/split/split_dimє
while/gru_cell_55/splitSplit*while/gru_cell_55/split/split_dim:output:0"while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/splitЫ
)while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_55/MatMul_1/ReadVariableOpМ
while/gru_cell_55/MatMul_1MatMulwhile_placeholder_21while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMul_1С
while/gru_cell_55/BiasAdd_1BiasAdd$while/gru_cell_55/MatMul_1:product:0"while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAdd_1
while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_55/Const_1
#while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_55/split_1/split_dim­
while/gru_cell_55/split_1SplitV$while/gru_cell_55/BiasAdd_1:output:0"while/gru_cell_55/Const_1:output:0,while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/split_1Џ
while/gru_cell_55/addAddV2 while/gru_cell_55/split:output:0"while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add
while/gru_cell_55/SigmoidSigmoidwhile/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/SigmoidГ
while/gru_cell_55/add_1AddV2 while/gru_cell_55/split:output:1"while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_1
while/gru_cell_55/Sigmoid_1Sigmoidwhile/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Sigmoid_1Ќ
while/gru_cell_55/mulMulwhile/gru_cell_55/Sigmoid_1:y:0"while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mulЊ
while/gru_cell_55/add_2AddV2 while/gru_cell_55/split:output:2while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_2
while/gru_cell_55/TanhTanhwhile/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Tanh
while/gru_cell_55/mul_1Mulwhile/gru_cell_55/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_1w
while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_55/sub/xЈ
while/gru_cell_55/subSub while/gru_cell_55/sub/x:output:0while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/subЂ
while/gru_cell_55/mul_2Mulwhile/gru_cell_55/sub:z:0while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_2Ї
while/gru_cell_55/add_3AddV2while/gru_cell_55/mul_1:z:0while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_55/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_55_matmul_1_readvariableop_resource4while_gru_cell_55_matmul_1_readvariableop_resource_0"f
0while_gru_cell_55_matmul_readvariableop_resource2while_gru_cell_55_matmul_readvariableop_resource_0"X
)while_gru_cell_55_readvariableop_resource+while_gru_cell_55_readvariableop_resource_0")
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
	
Ё
GRU_2_while_cond_1127925(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1127925___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1127925___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1127925___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1127925___redundant_placeholder3
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
џ!
т
while_body_1125344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_54_1125366_0
while_gru_cell_54_1125368_0
while_gru_cell_54_1125370_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_54_1125366
while_gru_cell_54_1125368
while_gru_cell_54_1125370Ђ)while/gru_cell_54/StatefulPartitionedCallУ
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
)while/gru_cell_54/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_54_1125366_0while_gru_cell_54_1125368_0while_gru_cell_54_1125370_0*
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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_11250452+
)while/gru_cell_54/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_54/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_54/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_54/StatefulPartitionedCall:output:1*^while/gru_cell_54/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_54_1125366while_gru_cell_54_1125366_0"8
while_gru_cell_54_1125368while_gru_cell_54_1125368_0"8
while_gru_cell_54_1125370while_gru_cell_54_1125370_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_54/StatefulPartitionedCall)while/gru_cell_54/StatefulPartitionedCall: 
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129605

inputs'
#gru_cell_55_readvariableop_resource.
*gru_cell_55_matmul_readvariableop_resource0
,gru_cell_55_matmul_1_readvariableop_resource
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
gru_cell_55/ReadVariableOpReadVariableOp#gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_55/ReadVariableOp
gru_cell_55/unstackUnpack"gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_55/unstackБ
!gru_cell_55/MatMul/ReadVariableOpReadVariableOp*gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_55/MatMul/ReadVariableOpЉ
gru_cell_55/MatMulMatMulstrided_slice_2:output:0)gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMulЃ
gru_cell_55/BiasAddBiasAddgru_cell_55/MatMul:product:0gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAddh
gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_55/Const
gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split/split_dimм
gru_cell_55/splitSplit$gru_cell_55/split/split_dim:output:0gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/splitЗ
#gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_55/MatMul_1/ReadVariableOpЅ
gru_cell_55/MatMul_1MatMulzeros:output:0+gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/MatMul_1Љ
gru_cell_55/BiasAdd_1BiasAddgru_cell_55/MatMul_1:product:0gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_55/BiasAdd_1
gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_55/Const_1
gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_55/split_1/split_dim
gru_cell_55/split_1SplitVgru_cell_55/BiasAdd_1:output:0gru_cell_55/Const_1:output:0&gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_55/split_1
gru_cell_55/addAddV2gru_cell_55/split:output:0gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add|
gru_cell_55/SigmoidSigmoidgru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid
gru_cell_55/add_1AddV2gru_cell_55/split:output:1gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_1
gru_cell_55/Sigmoid_1Sigmoidgru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Sigmoid_1
gru_cell_55/mulMulgru_cell_55/Sigmoid_1:y:0gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul
gru_cell_55/add_2AddV2gru_cell_55/split:output:2gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_2u
gru_cell_55/TanhTanhgru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/Tanh
gru_cell_55/mul_1Mulgru_cell_55/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_1k
gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_55/sub/x
gru_cell_55/subSubgru_cell_55/sub/x:output:0gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/sub
gru_cell_55/mul_2Mulgru_cell_55/sub:z:0gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/mul_2
gru_cell_55/add_3AddV2gru_cell_55/mul_1:z:0gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_55/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_55_readvariableop_resource*gru_cell_55_matmul_readvariableop_resource,gru_cell_55_matmul_1_readvariableop_resource*
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
while_body_1129515*
condR
while_cond_1129514*8
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
оH
и
GRU_1_while_body_1127388(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_54_readvariableop_resource_0<
8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_54_readvariableop_resource:
6gru_1_while_gru_cell_54_matmul_readvariableop_resource<
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_54/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_54/ReadVariableOpВ
GRU_1/while/gru_cell_54/unstackUnpack.GRU_1/while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_54/unstackз
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_54/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_54/MatMulг
GRU_1/while/gru_cell_54/BiasAddBiasAdd(GRU_1/while/gru_cell_54/MatMul:product:0(GRU_1/while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_54/BiasAdd
GRU_1/while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_54/Const
'GRU_1/while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_54/split/split_dim
GRU_1/while/gru_cell_54/splitSplit0GRU_1/while/gru_cell_54/split/split_dim:output:0(GRU_1/while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_54/splitн
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_54/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_54/MatMul_1й
!GRU_1/while/gru_cell_54/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_54/MatMul_1:product:0(GRU_1/while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_54/BiasAdd_1
GRU_1/while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_54/Const_1Ё
)GRU_1/while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_54/split_1/split_dimЫ
GRU_1/while/gru_cell_54/split_1SplitV*GRU_1/while/gru_cell_54/BiasAdd_1:output:0(GRU_1/while/gru_cell_54/Const_1:output:02GRU_1/while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_54/split_1Ч
GRU_1/while/gru_cell_54/addAddV2&GRU_1/while/gru_cell_54/split:output:0(GRU_1/while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add 
GRU_1/while/gru_cell_54/SigmoidSigmoidGRU_1/while/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_54/SigmoidЫ
GRU_1/while/gru_cell_54/add_1AddV2&GRU_1/while/gru_cell_54/split:output:1(GRU_1/while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_1І
!GRU_1/while/gru_cell_54/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_54/Sigmoid_1Ф
GRU_1/while/gru_cell_54/mulMul%GRU_1/while/gru_cell_54/Sigmoid_1:y:0(GRU_1/while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mulТ
GRU_1/while/gru_cell_54/add_2AddV2&GRU_1/while/gru_cell_54/split:output:2GRU_1/while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_2
GRU_1/while/gru_cell_54/TanhTanh!GRU_1/while/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/TanhЗ
GRU_1/while/gru_cell_54/mul_1Mul#GRU_1/while/gru_cell_54/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_1
GRU_1/while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_54/sub/xР
GRU_1/while/gru_cell_54/subSub&GRU_1/while/gru_cell_54/sub/x:output:0#GRU_1/while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/subК
GRU_1/while/gru_cell_54/mul_2MulGRU_1/while/gru_cell_54/sub:z:0 GRU_1/while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_2П
GRU_1/while/gru_cell_54/add_3AddV2!GRU_1/while/gru_cell_54/mul_1:z:0!GRU_1/while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_54/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resource:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_54_matmul_readvariableop_resource8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_54_readvariableop_resource1gru_1_while_gru_cell_54_readvariableop_resource_0"5
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
оH
и
GRU_2_while_body_1127202(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2'
#gru_2_while_gru_2_strided_slice_1_0c
_gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor_05
1gru_2_while_gru_cell_55_readvariableop_resource_0<
8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0>
:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0
gru_2_while_identity
gru_2_while_identity_1
gru_2_while_identity_2
gru_2_while_identity_3
gru_2_while_identity_4%
!gru_2_while_gru_2_strided_slice_1a
]gru_2_while_tensorarrayv2read_tensorlistgetitem_gru_2_tensorarrayunstack_tensorlistfromtensor3
/gru_2_while_gru_cell_55_readvariableop_resource:
6gru_2_while_gru_cell_55_matmul_readvariableop_resource<
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resourceЯ
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
&GRU_2/while/gru_cell_55/ReadVariableOpReadVariableOp1gru_2_while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_2/while/gru_cell_55/ReadVariableOpВ
GRU_2/while/gru_cell_55/unstackUnpack.GRU_2/while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_2/while/gru_cell_55/unstackз
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_2/while/gru_cell_55/MatMul/ReadVariableOpы
GRU_2/while/gru_cell_55/MatMulMatMul6GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_2/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_2/while/gru_cell_55/MatMulг
GRU_2/while/gru_cell_55/BiasAddBiasAdd(GRU_2/while/gru_cell_55/MatMul:product:0(GRU_2/while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_2/while/gru_cell_55/BiasAdd
GRU_2/while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_2/while/gru_cell_55/Const
'GRU_2/while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_2/while/gru_cell_55/split/split_dim
GRU_2/while/gru_cell_55/splitSplit0GRU_2/while/gru_cell_55/split/split_dim:output:0(GRU_2/while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_2/while/gru_cell_55/splitн
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpд
 GRU_2/while/gru_cell_55/MatMul_1MatMulgru_2_while_placeholder_27GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_2/while/gru_cell_55/MatMul_1й
!GRU_2/while/gru_cell_55/BiasAdd_1BiasAdd*GRU_2/while/gru_cell_55/MatMul_1:product:0(GRU_2/while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_2/while/gru_cell_55/BiasAdd_1
GRU_2/while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_2/while/gru_cell_55/Const_1Ё
)GRU_2/while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_2/while/gru_cell_55/split_1/split_dimЫ
GRU_2/while/gru_cell_55/split_1SplitV*GRU_2/while/gru_cell_55/BiasAdd_1:output:0(GRU_2/while/gru_cell_55/Const_1:output:02GRU_2/while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_2/while/gru_cell_55/split_1Ч
GRU_2/while/gru_cell_55/addAddV2&GRU_2/while/gru_cell_55/split:output:0(GRU_2/while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add 
GRU_2/while/gru_cell_55/SigmoidSigmoidGRU_2/while/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_2/while/gru_cell_55/SigmoidЫ
GRU_2/while/gru_cell_55/add_1AddV2&GRU_2/while/gru_cell_55/split:output:1(GRU_2/while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_1І
!GRU_2/while/gru_cell_55/Sigmoid_1Sigmoid!GRU_2/while/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_2/while/gru_cell_55/Sigmoid_1Ф
GRU_2/while/gru_cell_55/mulMul%GRU_2/while/gru_cell_55/Sigmoid_1:y:0(GRU_2/while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mulТ
GRU_2/while/gru_cell_55/add_2AddV2&GRU_2/while/gru_cell_55/split:output:2GRU_2/while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_2
GRU_2/while/gru_cell_55/TanhTanh!GRU_2/while/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/TanhЗ
GRU_2/while/gru_cell_55/mul_1Mul#GRU_2/while/gru_cell_55/Sigmoid:y:0gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_1
GRU_2/while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_2/while/gru_cell_55/sub/xР
GRU_2/while/gru_cell_55/subSub&GRU_2/while/gru_cell_55/sub/x:output:0#GRU_2/while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/subК
GRU_2/while/gru_cell_55/mul_2MulGRU_2/while/gru_cell_55/sub:z:0 GRU_2/while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/mul_2П
GRU_2/while/gru_cell_55/add_3AddV2!GRU_2/while/gru_cell_55/mul_1:z:0!GRU_2/while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/gru_cell_55/add_3§
0GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_2_while_placeholder_1gru_2_while_placeholder!GRU_2/while/gru_cell_55/add_3:z:0*
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
GRU_2/while/Identity_4Identity!GRU_2/while/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_2/while/Identity_4"H
!gru_2_while_gru_2_strided_slice_1#gru_2_while_gru_2_strided_slice_1_0"v
8gru_2_while_gru_cell_55_matmul_1_readvariableop_resource:gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0"r
6gru_2_while_gru_cell_55_matmul_readvariableop_resource8gru_2_while_gru_cell_55_matmul_readvariableop_resource_0"d
/gru_2_while_gru_cell_55_readvariableop_resource1gru_2_while_gru_cell_55_readvariableop_resource_0"5
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
while_body_1126024
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_55_1126046_0
while_gru_cell_55_1126048_0
while_gru_cell_55_1126050_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_55_1126046
while_gru_cell_55_1126048
while_gru_cell_55_1126050Ђ)while/gru_cell_55/StatefulPartitionedCallУ
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
)while/gru_cell_55/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_55_1126046_0while_gru_cell_55_1126048_0while_gru_cell_55_1126050_0*
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_11256472+
)while/gru_cell_55/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_55/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_55/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/gru_cell_55/StatefulPartitionedCall:output:1*^while/gru_cell_55/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"8
while_gru_cell_55_1126046while_gru_cell_55_1126046_0"8
while_gru_cell_55_1126048while_gru_cell_55_1126048_0"8
while_gru_cell_55_1126050while_gru_cell_55_1126050_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ: : :::2V
)while/gru_cell_55/StatefulPartitionedCall)while/gru_cell_55/StatefulPartitionedCall: 
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
while_cond_1126327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1126327___redundant_placeholder05
1while_while_cond_1126327___redundant_placeholder15
1while_while_cond_1126327___redundant_placeholder25
1while_while_cond_1126327___redundant_placeholder3
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
GRU_1_while_body_1128112(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_54_readvariableop_resource_0<
8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_54_readvariableop_resource:
6gru_1_while_gru_cell_54_matmul_readvariableop_resource<
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_54/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_54/ReadVariableOpВ
GRU_1/while/gru_cell_54/unstackUnpack.GRU_1/while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_54/unstackз
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_54/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_54/MatMulг
GRU_1/while/gru_cell_54/BiasAddBiasAdd(GRU_1/while/gru_cell_54/MatMul:product:0(GRU_1/while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_54/BiasAdd
GRU_1/while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_54/Const
'GRU_1/while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_54/split/split_dim
GRU_1/while/gru_cell_54/splitSplit0GRU_1/while/gru_cell_54/split/split_dim:output:0(GRU_1/while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_54/splitн
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_54/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_54/MatMul_1й
!GRU_1/while/gru_cell_54/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_54/MatMul_1:product:0(GRU_1/while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_54/BiasAdd_1
GRU_1/while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_54/Const_1Ё
)GRU_1/while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_54/split_1/split_dimЫ
GRU_1/while/gru_cell_54/split_1SplitV*GRU_1/while/gru_cell_54/BiasAdd_1:output:0(GRU_1/while/gru_cell_54/Const_1:output:02GRU_1/while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_54/split_1Ч
GRU_1/while/gru_cell_54/addAddV2&GRU_1/while/gru_cell_54/split:output:0(GRU_1/while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add 
GRU_1/while/gru_cell_54/SigmoidSigmoidGRU_1/while/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_54/SigmoidЫ
GRU_1/while/gru_cell_54/add_1AddV2&GRU_1/while/gru_cell_54/split:output:1(GRU_1/while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_1І
!GRU_1/while/gru_cell_54/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_54/Sigmoid_1Ф
GRU_1/while/gru_cell_54/mulMul%GRU_1/while/gru_cell_54/Sigmoid_1:y:0(GRU_1/while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mulТ
GRU_1/while/gru_cell_54/add_2AddV2&GRU_1/while/gru_cell_54/split:output:2GRU_1/while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_2
GRU_1/while/gru_cell_54/TanhTanh!GRU_1/while/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/TanhЗ
GRU_1/while/gru_cell_54/mul_1Mul#GRU_1/while/gru_cell_54/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_1
GRU_1/while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_54/sub/xР
GRU_1/while/gru_cell_54/subSub&GRU_1/while/gru_cell_54/sub/x:output:0#GRU_1/while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/subК
GRU_1/while/gru_cell_54/mul_2MulGRU_1/while/gru_cell_54/sub:z:0 GRU_1/while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_2П
GRU_1/while/gru_cell_54/add_3AddV2!GRU_1/while/gru_cell_54/mul_1:z:0!GRU_1/while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_54/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resource:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_54_matmul_readvariableop_resource8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_54_readvariableop_resource1gru_1_while_gru_cell_54_readvariableop_resource_0"5
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
@
Ж
while_body_1129515
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_55_readvariableop_resource_06
2while_gru_cell_55_matmul_readvariableop_resource_08
4while_gru_cell_55_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_55_readvariableop_resource4
0while_gru_cell_55_matmul_readvariableop_resource6
2while_gru_cell_55_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_55/ReadVariableOpReadVariableOp+while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_55/ReadVariableOp 
while/gru_cell_55/unstackUnpack(while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_55/unstackХ
'while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_55/MatMul/ReadVariableOpг
while/gru_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMulЛ
while/gru_cell_55/BiasAddBiasAdd"while/gru_cell_55/MatMul:product:0"while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAddt
while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_55/Const
!while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_55/split/split_dimє
while/gru_cell_55/splitSplit*while/gru_cell_55/split/split_dim:output:0"while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/splitЫ
)while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_55/MatMul_1/ReadVariableOpМ
while/gru_cell_55/MatMul_1MatMulwhile_placeholder_21while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMul_1С
while/gru_cell_55/BiasAdd_1BiasAdd$while/gru_cell_55/MatMul_1:product:0"while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAdd_1
while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_55/Const_1
#while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_55/split_1/split_dim­
while/gru_cell_55/split_1SplitV$while/gru_cell_55/BiasAdd_1:output:0"while/gru_cell_55/Const_1:output:0,while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/split_1Џ
while/gru_cell_55/addAddV2 while/gru_cell_55/split:output:0"while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add
while/gru_cell_55/SigmoidSigmoidwhile/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/SigmoidГ
while/gru_cell_55/add_1AddV2 while/gru_cell_55/split:output:1"while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_1
while/gru_cell_55/Sigmoid_1Sigmoidwhile/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Sigmoid_1Ќ
while/gru_cell_55/mulMulwhile/gru_cell_55/Sigmoid_1:y:0"while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mulЊ
while/gru_cell_55/add_2AddV2 while/gru_cell_55/split:output:2while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_2
while/gru_cell_55/TanhTanhwhile/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Tanh
while/gru_cell_55/mul_1Mulwhile/gru_cell_55/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_1w
while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_55/sub/xЈ
while/gru_cell_55/subSub while/gru_cell_55/sub/x:output:0while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/subЂ
while/gru_cell_55/mul_2Mulwhile/gru_cell_55/sub:z:0while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_2Ї
while/gru_cell_55/add_3AddV2while/gru_cell_55/mul_1:z:0while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_55/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_55_matmul_1_readvariableop_resource4while_gru_cell_55_matmul_1_readvariableop_resource_0"f
0while_gru_cell_55_matmul_readvariableop_resource2while_gru_cell_55_matmul_readvariableop_resource_0"X
)while_gru_cell_55_readvariableop_resource+while_gru_cell_55_readvariableop_resource_0")
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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1129906

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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1126088

inputs
gru_cell_55_1126012
gru_cell_55_1126014
gru_cell_55_1126016
identityЂ#gru_cell_55/StatefulPartitionedCallЂwhileD
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
#gru_cell_55/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_55_1126012gru_cell_55_1126014gru_cell_55_1126016*
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_11256472%
#gru_cell_55/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_55_1126012gru_cell_55_1126014gru_cell_55_1126016*
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
while_body_1126024*
condR
while_cond_1126023*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_55/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_55/StatefulPartitionedCall#gru_cell_55/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЊX


#Supervisor_GRU_2_while_body_1124856>
:supervisor_gru_2_while_supervisor_gru_2_while_loop_counterD
@supervisor_gru_2_while_supervisor_gru_2_while_maximum_iterations&
"supervisor_gru_2_while_placeholder(
$supervisor_gru_2_while_placeholder_1(
$supervisor_gru_2_while_placeholder_2=
9supervisor_gru_2_while_supervisor_gru_2_strided_slice_1_0y
usupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor_0@
<supervisor_gru_2_while_gru_cell_55_readvariableop_resource_0G
Csupervisor_gru_2_while_gru_cell_55_matmul_readvariableop_resource_0I
Esupervisor_gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0#
supervisor_gru_2_while_identity%
!supervisor_gru_2_while_identity_1%
!supervisor_gru_2_while_identity_2%
!supervisor_gru_2_while_identity_3%
!supervisor_gru_2_while_identity_4;
7supervisor_gru_2_while_supervisor_gru_2_strided_slice_1w
ssupervisor_gru_2_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_2_tensorarrayunstack_tensorlistfromtensor>
:supervisor_gru_2_while_gru_cell_55_readvariableop_resourceE
Asupervisor_gru_2_while_gru_cell_55_matmul_readvariableop_resourceG
Csupervisor_gru_2_while_gru_cell_55_matmul_1_readvariableop_resourceх
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
1Supervisor/GRU_2/while/gru_cell_55/ReadVariableOpReadVariableOp<supervisor_gru_2_while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype023
1Supervisor/GRU_2/while/gru_cell_55/ReadVariableOpг
*Supervisor/GRU_2/while/gru_cell_55/unstackUnpack9Supervisor/GRU_2/while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2,
*Supervisor/GRU_2/while/gru_cell_55/unstackј
8Supervisor/GRU_2/while/gru_cell_55/MatMul/ReadVariableOpReadVariableOpCsupervisor_gru_2_while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02:
8Supervisor/GRU_2/while/gru_cell_55/MatMul/ReadVariableOp
)Supervisor/GRU_2/while/gru_cell_55/MatMulMatMulASupervisor/GRU_2/while/TensorArrayV2Read/TensorListGetItem:item:0@Supervisor/GRU_2/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2+
)Supervisor/GRU_2/while/gru_cell_55/MatMulџ
*Supervisor/GRU_2/while/gru_cell_55/BiasAddBiasAdd3Supervisor/GRU_2/while/gru_cell_55/MatMul:product:03Supervisor/GRU_2/while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2,
*Supervisor/GRU_2/while/gru_cell_55/BiasAdd
(Supervisor/GRU_2/while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(Supervisor/GRU_2/while/gru_cell_55/ConstГ
2Supervisor/GRU_2/while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2Supervisor/GRU_2/while/gru_cell_55/split/split_dimИ
(Supervisor/GRU_2/while/gru_cell_55/splitSplit;Supervisor/GRU_2/while/gru_cell_55/split/split_dim:output:03Supervisor/GRU_2/while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2*
(Supervisor/GRU_2/while/gru_cell_55/splitў
:Supervisor/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOpEsupervisor_gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02<
:Supervisor/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOp
+Supervisor/GRU_2/while/gru_cell_55/MatMul_1MatMul$supervisor_gru_2_while_placeholder_2BSupervisor/GRU_2/while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2-
+Supervisor/GRU_2/while/gru_cell_55/MatMul_1
,Supervisor/GRU_2/while/gru_cell_55/BiasAdd_1BiasAdd5Supervisor/GRU_2/while/gru_cell_55/MatMul_1:product:03Supervisor/GRU_2/while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2.
,Supervisor/GRU_2/while/gru_cell_55/BiasAdd_1­
*Supervisor/GRU_2/while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2,
*Supervisor/GRU_2/while/gru_cell_55/Const_1З
4Supervisor/GRU_2/while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ26
4Supervisor/GRU_2/while/gru_cell_55/split_1/split_dim
*Supervisor/GRU_2/while/gru_cell_55/split_1SplitV5Supervisor/GRU_2/while/gru_cell_55/BiasAdd_1:output:03Supervisor/GRU_2/while/gru_cell_55/Const_1:output:0=Supervisor/GRU_2/while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*Supervisor/GRU_2/while/gru_cell_55/split_1ѓ
&Supervisor/GRU_2/while/gru_cell_55/addAddV21Supervisor/GRU_2/while/gru_cell_55/split:output:03Supervisor/GRU_2/while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_55/addС
*Supervisor/GRU_2/while/gru_cell_55/SigmoidSigmoid*Supervisor/GRU_2/while/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*Supervisor/GRU_2/while/gru_cell_55/Sigmoidї
(Supervisor/GRU_2/while/gru_cell_55/add_1AddV21Supervisor/GRU_2/while/gru_cell_55/split:output:13Supervisor/GRU_2/while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_55/add_1Ч
,Supervisor/GRU_2/while/gru_cell_55/Sigmoid_1Sigmoid,Supervisor/GRU_2/while/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,Supervisor/GRU_2/while/gru_cell_55/Sigmoid_1№
&Supervisor/GRU_2/while/gru_cell_55/mulMul0Supervisor/GRU_2/while/gru_cell_55/Sigmoid_1:y:03Supervisor/GRU_2/while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_55/mulю
(Supervisor/GRU_2/while/gru_cell_55/add_2AddV21Supervisor/GRU_2/while/gru_cell_55/split:output:2*Supervisor/GRU_2/while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_55/add_2К
'Supervisor/GRU_2/while/gru_cell_55/TanhTanh,Supervisor/GRU_2/while/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'Supervisor/GRU_2/while/gru_cell_55/Tanhу
(Supervisor/GRU_2/while/gru_cell_55/mul_1Mul.Supervisor/GRU_2/while/gru_cell_55/Sigmoid:y:0$supervisor_gru_2_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_55/mul_1
(Supervisor/GRU_2/while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(Supervisor/GRU_2/while/gru_cell_55/sub/xь
&Supervisor/GRU_2/while/gru_cell_55/subSub1Supervisor/GRU_2/while/gru_cell_55/sub/x:output:0.Supervisor/GRU_2/while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/while/gru_cell_55/subц
(Supervisor/GRU_2/while/gru_cell_55/mul_2Mul*Supervisor/GRU_2/while/gru_cell_55/sub:z:0+Supervisor/GRU_2/while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_55/mul_2ы
(Supervisor/GRU_2/while/gru_cell_55/add_3AddV2,Supervisor/GRU_2/while/gru_cell_55/mul_1:z:0,Supervisor/GRU_2/while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_2/while/gru_cell_55/add_3Д
;Supervisor/GRU_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$supervisor_gru_2_while_placeholder_1"supervisor_gru_2_while_placeholder,Supervisor/GRU_2/while/gru_cell_55/add_3:z:0*
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
!Supervisor/GRU_2/while/Identity_4Identity,Supervisor/GRU_2/while/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_2/while/Identity_4"
Csupervisor_gru_2_while_gru_cell_55_matmul_1_readvariableop_resourceEsupervisor_gru_2_while_gru_cell_55_matmul_1_readvariableop_resource_0"
Asupervisor_gru_2_while_gru_cell_55_matmul_readvariableop_resourceCsupervisor_gru_2_while_gru_cell_55_matmul_readvariableop_resource_0"z
:supervisor_gru_2_while_gru_cell_55_readvariableop_resource<supervisor_gru_2_while_gru_cell_55_readvariableop_resource_0"K
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
р	
Џ
-__inference_gru_cell_55_layer_call_fn_1130042

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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_11256472
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


#Supervisor_GRU_1_while_body_1124701>
:supervisor_gru_1_while_supervisor_gru_1_while_loop_counterD
@supervisor_gru_1_while_supervisor_gru_1_while_maximum_iterations&
"supervisor_gru_1_while_placeholder(
$supervisor_gru_1_while_placeholder_1(
$supervisor_gru_1_while_placeholder_2=
9supervisor_gru_1_while_supervisor_gru_1_strided_slice_1_0y
usupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor_0@
<supervisor_gru_1_while_gru_cell_54_readvariableop_resource_0G
Csupervisor_gru_1_while_gru_cell_54_matmul_readvariableop_resource_0I
Esupervisor_gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0#
supervisor_gru_1_while_identity%
!supervisor_gru_1_while_identity_1%
!supervisor_gru_1_while_identity_2%
!supervisor_gru_1_while_identity_3%
!supervisor_gru_1_while_identity_4;
7supervisor_gru_1_while_supervisor_gru_1_strided_slice_1w
ssupervisor_gru_1_while_tensorarrayv2read_tensorlistgetitem_supervisor_gru_1_tensorarrayunstack_tensorlistfromtensor>
:supervisor_gru_1_while_gru_cell_54_readvariableop_resourceE
Asupervisor_gru_1_while_gru_cell_54_matmul_readvariableop_resourceG
Csupervisor_gru_1_while_gru_cell_54_matmul_1_readvariableop_resourceх
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
1Supervisor/GRU_1/while/gru_cell_54/ReadVariableOpReadVariableOp<supervisor_gru_1_while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype023
1Supervisor/GRU_1/while/gru_cell_54/ReadVariableOpг
*Supervisor/GRU_1/while/gru_cell_54/unstackUnpack9Supervisor/GRU_1/while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2,
*Supervisor/GRU_1/while/gru_cell_54/unstackј
8Supervisor/GRU_1/while/gru_cell_54/MatMul/ReadVariableOpReadVariableOpCsupervisor_gru_1_while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02:
8Supervisor/GRU_1/while/gru_cell_54/MatMul/ReadVariableOp
)Supervisor/GRU_1/while/gru_cell_54/MatMulMatMulASupervisor/GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:0@Supervisor/GRU_1/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2+
)Supervisor/GRU_1/while/gru_cell_54/MatMulџ
*Supervisor/GRU_1/while/gru_cell_54/BiasAddBiasAdd3Supervisor/GRU_1/while/gru_cell_54/MatMul:product:03Supervisor/GRU_1/while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2,
*Supervisor/GRU_1/while/gru_cell_54/BiasAdd
(Supervisor/GRU_1/while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(Supervisor/GRU_1/while/gru_cell_54/ConstГ
2Supervisor/GRU_1/while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2Supervisor/GRU_1/while/gru_cell_54/split/split_dimИ
(Supervisor/GRU_1/while/gru_cell_54/splitSplit;Supervisor/GRU_1/while/gru_cell_54/split/split_dim:output:03Supervisor/GRU_1/while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2*
(Supervisor/GRU_1/while/gru_cell_54/splitў
:Supervisor/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOpEsupervisor_gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02<
:Supervisor/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOp
+Supervisor/GRU_1/while/gru_cell_54/MatMul_1MatMul$supervisor_gru_1_while_placeholder_2BSupervisor/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2-
+Supervisor/GRU_1/while/gru_cell_54/MatMul_1
,Supervisor/GRU_1/while/gru_cell_54/BiasAdd_1BiasAdd5Supervisor/GRU_1/while/gru_cell_54/MatMul_1:product:03Supervisor/GRU_1/while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2.
,Supervisor/GRU_1/while/gru_cell_54/BiasAdd_1­
*Supervisor/GRU_1/while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2,
*Supervisor/GRU_1/while/gru_cell_54/Const_1З
4Supervisor/GRU_1/while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ26
4Supervisor/GRU_1/while/gru_cell_54/split_1/split_dim
*Supervisor/GRU_1/while/gru_cell_54/split_1SplitV5Supervisor/GRU_1/while/gru_cell_54/BiasAdd_1:output:03Supervisor/GRU_1/while/gru_cell_54/Const_1:output:0=Supervisor/GRU_1/while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*Supervisor/GRU_1/while/gru_cell_54/split_1ѓ
&Supervisor/GRU_1/while/gru_cell_54/addAddV21Supervisor/GRU_1/while/gru_cell_54/split:output:03Supervisor/GRU_1/while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_54/addС
*Supervisor/GRU_1/while/gru_cell_54/SigmoidSigmoid*Supervisor/GRU_1/while/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*Supervisor/GRU_1/while/gru_cell_54/Sigmoidї
(Supervisor/GRU_1/while/gru_cell_54/add_1AddV21Supervisor/GRU_1/while/gru_cell_54/split:output:13Supervisor/GRU_1/while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_54/add_1Ч
,Supervisor/GRU_1/while/gru_cell_54/Sigmoid_1Sigmoid,Supervisor/GRU_1/while/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,Supervisor/GRU_1/while/gru_cell_54/Sigmoid_1№
&Supervisor/GRU_1/while/gru_cell_54/mulMul0Supervisor/GRU_1/while/gru_cell_54/Sigmoid_1:y:03Supervisor/GRU_1/while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_54/mulю
(Supervisor/GRU_1/while/gru_cell_54/add_2AddV21Supervisor/GRU_1/while/gru_cell_54/split:output:2*Supervisor/GRU_1/while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_54/add_2К
'Supervisor/GRU_1/while/gru_cell_54/TanhTanh,Supervisor/GRU_1/while/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2)
'Supervisor/GRU_1/while/gru_cell_54/Tanhу
(Supervisor/GRU_1/while/gru_cell_54/mul_1Mul.Supervisor/GRU_1/while/gru_cell_54/Sigmoid:y:0$supervisor_gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_54/mul_1
(Supervisor/GRU_1/while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(Supervisor/GRU_1/while/gru_cell_54/sub/xь
&Supervisor/GRU_1/while/gru_cell_54/subSub1Supervisor/GRU_1/while/gru_cell_54/sub/x:output:0.Supervisor/GRU_1/while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/while/gru_cell_54/subц
(Supervisor/GRU_1/while/gru_cell_54/mul_2Mul*Supervisor/GRU_1/while/gru_cell_54/sub:z:0+Supervisor/GRU_1/while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_54/mul_2ы
(Supervisor/GRU_1/while/gru_cell_54/add_3AddV2,Supervisor/GRU_1/while/gru_cell_54/mul_1:z:0,Supervisor/GRU_1/while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(Supervisor/GRU_1/while/gru_cell_54/add_3Д
;Supervisor/GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$supervisor_gru_1_while_placeholder_1"supervisor_gru_1_while_placeholder,Supervisor/GRU_1/while/gru_cell_54/add_3:z:0*
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
!Supervisor/GRU_1/while/Identity_4Identity,Supervisor/GRU_1/while/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_1/while/Identity_4"
Csupervisor_gru_1_while_gru_cell_54_matmul_1_readvariableop_resourceEsupervisor_gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0"
Asupervisor_gru_1_while_gru_cell_54_matmul_readvariableop_resourceCsupervisor_gru_1_while_gru_cell_54_matmul_readvariableop_resource_0"z
:supervisor_gru_1_while_gru_cell_54_readvariableop_resource<supervisor_gru_1_while_gru_cell_54_readvariableop_resource_0"K
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
я
ь
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1129974

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
Ц
о
"__inference__wrapped_model_1124973
gru_1_input8
4supervisor_gru_1_gru_cell_54_readvariableop_resource?
;supervisor_gru_1_gru_cell_54_matmul_readvariableop_resourceA
=supervisor_gru_1_gru_cell_54_matmul_1_readvariableop_resource8
4supervisor_gru_2_gru_cell_55_readvariableop_resource?
;supervisor_gru_2_gru_cell_55_matmul_readvariableop_resourceA
=supervisor_gru_2_gru_cell_55_matmul_1_readvariableop_resource4
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
+Supervisor/GRU_1/gru_cell_54/ReadVariableOpReadVariableOp4supervisor_gru_1_gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02-
+Supervisor/GRU_1/gru_cell_54/ReadVariableOpС
$Supervisor/GRU_1/gru_cell_54/unstackUnpack3Supervisor/GRU_1/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2&
$Supervisor/GRU_1/gru_cell_54/unstackф
2Supervisor/GRU_1/gru_cell_54/MatMul/ReadVariableOpReadVariableOp;supervisor_gru_1_gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype024
2Supervisor/GRU_1/gru_cell_54/MatMul/ReadVariableOpэ
#Supervisor/GRU_1/gru_cell_54/MatMulMatMul)Supervisor/GRU_1/strided_slice_2:output:0:Supervisor/GRU_1/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2%
#Supervisor/GRU_1/gru_cell_54/MatMulч
$Supervisor/GRU_1/gru_cell_54/BiasAddBiasAdd-Supervisor/GRU_1/gru_cell_54/MatMul:product:0-Supervisor/GRU_1/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2&
$Supervisor/GRU_1/gru_cell_54/BiasAdd
"Supervisor/GRU_1/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"Supervisor/GRU_1/gru_cell_54/ConstЇ
,Supervisor/GRU_1/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_1/gru_cell_54/split/split_dim 
"Supervisor/GRU_1/gru_cell_54/splitSplit5Supervisor/GRU_1/gru_cell_54/split/split_dim:output:0-Supervisor/GRU_1/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2$
"Supervisor/GRU_1/gru_cell_54/splitъ
4Supervisor/GRU_1/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp=supervisor_gru_1_gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype026
4Supervisor/GRU_1/gru_cell_54/MatMul_1/ReadVariableOpщ
%Supervisor/GRU_1/gru_cell_54/MatMul_1MatMulSupervisor/GRU_1/zeros:output:0<Supervisor/GRU_1/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2'
%Supervisor/GRU_1/gru_cell_54/MatMul_1э
&Supervisor/GRU_1/gru_cell_54/BiasAdd_1BiasAdd/Supervisor/GRU_1/gru_cell_54/MatMul_1:product:0-Supervisor/GRU_1/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2(
&Supervisor/GRU_1/gru_cell_54/BiasAdd_1Ё
$Supervisor/GRU_1/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2&
$Supervisor/GRU_1/gru_cell_54/Const_1Ћ
.Supervisor/GRU_1/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.Supervisor/GRU_1/gru_cell_54/split_1/split_dimф
$Supervisor/GRU_1/gru_cell_54/split_1SplitV/Supervisor/GRU_1/gru_cell_54/BiasAdd_1:output:0-Supervisor/GRU_1/gru_cell_54/Const_1:output:07Supervisor/GRU_1/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2&
$Supervisor/GRU_1/gru_cell_54/split_1л
 Supervisor/GRU_1/gru_cell_54/addAddV2+Supervisor/GRU_1/gru_cell_54/split:output:0-Supervisor/GRU_1/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_54/addЏ
$Supervisor/GRU_1/gru_cell_54/SigmoidSigmoid$Supervisor/GRU_1/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$Supervisor/GRU_1/gru_cell_54/Sigmoidп
"Supervisor/GRU_1/gru_cell_54/add_1AddV2+Supervisor/GRU_1/gru_cell_54/split:output:1-Supervisor/GRU_1/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_54/add_1Е
&Supervisor/GRU_1/gru_cell_54/Sigmoid_1Sigmoid&Supervisor/GRU_1/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_1/gru_cell_54/Sigmoid_1и
 Supervisor/GRU_1/gru_cell_54/mulMul*Supervisor/GRU_1/gru_cell_54/Sigmoid_1:y:0-Supervisor/GRU_1/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_54/mulж
"Supervisor/GRU_1/gru_cell_54/add_2AddV2+Supervisor/GRU_1/gru_cell_54/split:output:2$Supervisor/GRU_1/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_54/add_2Ј
!Supervisor/GRU_1/gru_cell_54/TanhTanh&Supervisor/GRU_1/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_1/gru_cell_54/TanhЬ
"Supervisor/GRU_1/gru_cell_54/mul_1Mul(Supervisor/GRU_1/gru_cell_54/Sigmoid:y:0Supervisor/GRU_1/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_54/mul_1
"Supervisor/GRU_1/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"Supervisor/GRU_1/gru_cell_54/sub/xд
 Supervisor/GRU_1/gru_cell_54/subSub+Supervisor/GRU_1/gru_cell_54/sub/x:output:0(Supervisor/GRU_1/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_1/gru_cell_54/subЮ
"Supervisor/GRU_1/gru_cell_54/mul_2Mul$Supervisor/GRU_1/gru_cell_54/sub:z:0%Supervisor/GRU_1/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_54/mul_2г
"Supervisor/GRU_1/gru_cell_54/add_3AddV2&Supervisor/GRU_1/gru_cell_54/mul_1:z:0&Supervisor/GRU_1/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_1/gru_cell_54/add_3Б
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
Supervisor/GRU_1/whileWhile,Supervisor/GRU_1/while/loop_counter:output:02Supervisor/GRU_1/while/maximum_iterations:output:0Supervisor/GRU_1/time:output:0)Supervisor/GRU_1/TensorArrayV2_1:handle:0Supervisor/GRU_1/zeros:output:0)Supervisor/GRU_1/strided_slice_1:output:0HSupervisor/GRU_1/TensorArrayUnstack/TensorListFromTensor:output_handle:04supervisor_gru_1_gru_cell_54_readvariableop_resource;supervisor_gru_1_gru_cell_54_matmul_readvariableop_resource=supervisor_gru_1_gru_cell_54_matmul_1_readvariableop_resource*
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
#Supervisor_GRU_1_while_body_1124701*/
cond'R%
#Supervisor_GRU_1_while_cond_1124700*8
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
+Supervisor/GRU_2/gru_cell_55/ReadVariableOpReadVariableOp4supervisor_gru_2_gru_cell_55_readvariableop_resource*
_output_shapes

:<*
dtype02-
+Supervisor/GRU_2/gru_cell_55/ReadVariableOpС
$Supervisor/GRU_2/gru_cell_55/unstackUnpack3Supervisor/GRU_2/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2&
$Supervisor/GRU_2/gru_cell_55/unstackф
2Supervisor/GRU_2/gru_cell_55/MatMul/ReadVariableOpReadVariableOp;supervisor_gru_2_gru_cell_55_matmul_readvariableop_resource*
_output_shapes

:<*
dtype024
2Supervisor/GRU_2/gru_cell_55/MatMul/ReadVariableOpэ
#Supervisor/GRU_2/gru_cell_55/MatMulMatMul)Supervisor/GRU_2/strided_slice_2:output:0:Supervisor/GRU_2/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2%
#Supervisor/GRU_2/gru_cell_55/MatMulч
$Supervisor/GRU_2/gru_cell_55/BiasAddBiasAdd-Supervisor/GRU_2/gru_cell_55/MatMul:product:0-Supervisor/GRU_2/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2&
$Supervisor/GRU_2/gru_cell_55/BiasAdd
"Supervisor/GRU_2/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"Supervisor/GRU_2/gru_cell_55/ConstЇ
,Supervisor/GRU_2/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2.
,Supervisor/GRU_2/gru_cell_55/split/split_dim 
"Supervisor/GRU_2/gru_cell_55/splitSplit5Supervisor/GRU_2/gru_cell_55/split/split_dim:output:0-Supervisor/GRU_2/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2$
"Supervisor/GRU_2/gru_cell_55/splitъ
4Supervisor/GRU_2/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp=supervisor_gru_2_gru_cell_55_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype026
4Supervisor/GRU_2/gru_cell_55/MatMul_1/ReadVariableOpщ
%Supervisor/GRU_2/gru_cell_55/MatMul_1MatMulSupervisor/GRU_2/zeros:output:0<Supervisor/GRU_2/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2'
%Supervisor/GRU_2/gru_cell_55/MatMul_1э
&Supervisor/GRU_2/gru_cell_55/BiasAdd_1BiasAdd/Supervisor/GRU_2/gru_cell_55/MatMul_1:product:0-Supervisor/GRU_2/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2(
&Supervisor/GRU_2/gru_cell_55/BiasAdd_1Ё
$Supervisor/GRU_2/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2&
$Supervisor/GRU_2/gru_cell_55/Const_1Ћ
.Supervisor/GRU_2/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.Supervisor/GRU_2/gru_cell_55/split_1/split_dimф
$Supervisor/GRU_2/gru_cell_55/split_1SplitV/Supervisor/GRU_2/gru_cell_55/BiasAdd_1:output:0-Supervisor/GRU_2/gru_cell_55/Const_1:output:07Supervisor/GRU_2/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2&
$Supervisor/GRU_2/gru_cell_55/split_1л
 Supervisor/GRU_2/gru_cell_55/addAddV2+Supervisor/GRU_2/gru_cell_55/split:output:0-Supervisor/GRU_2/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_55/addЏ
$Supervisor/GRU_2/gru_cell_55/SigmoidSigmoid$Supervisor/GRU_2/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$Supervisor/GRU_2/gru_cell_55/Sigmoidп
"Supervisor/GRU_2/gru_cell_55/add_1AddV2+Supervisor/GRU_2/gru_cell_55/split:output:1-Supervisor/GRU_2/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_55/add_1Е
&Supervisor/GRU_2/gru_cell_55/Sigmoid_1Sigmoid&Supervisor/GRU_2/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&Supervisor/GRU_2/gru_cell_55/Sigmoid_1и
 Supervisor/GRU_2/gru_cell_55/mulMul*Supervisor/GRU_2/gru_cell_55/Sigmoid_1:y:0-Supervisor/GRU_2/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_55/mulж
"Supervisor/GRU_2/gru_cell_55/add_2AddV2+Supervisor/GRU_2/gru_cell_55/split:output:2$Supervisor/GRU_2/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_55/add_2Ј
!Supervisor/GRU_2/gru_cell_55/TanhTanh&Supervisor/GRU_2/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!Supervisor/GRU_2/gru_cell_55/TanhЬ
"Supervisor/GRU_2/gru_cell_55/mul_1Mul(Supervisor/GRU_2/gru_cell_55/Sigmoid:y:0Supervisor/GRU_2/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_55/mul_1
"Supervisor/GRU_2/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"Supervisor/GRU_2/gru_cell_55/sub/xд
 Supervisor/GRU_2/gru_cell_55/subSub+Supervisor/GRU_2/gru_cell_55/sub/x:output:0(Supervisor/GRU_2/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 Supervisor/GRU_2/gru_cell_55/subЮ
"Supervisor/GRU_2/gru_cell_55/mul_2Mul$Supervisor/GRU_2/gru_cell_55/sub:z:0%Supervisor/GRU_2/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_55/mul_2г
"Supervisor/GRU_2/gru_cell_55/add_3AddV2&Supervisor/GRU_2/gru_cell_55/mul_1:z:0&Supervisor/GRU_2/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Supervisor/GRU_2/gru_cell_55/add_3Б
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
Supervisor/GRU_2/whileWhile,Supervisor/GRU_2/while/loop_counter:output:02Supervisor/GRU_2/while/maximum_iterations:output:0Supervisor/GRU_2/time:output:0)Supervisor/GRU_2/TensorArrayV2_1:handle:0Supervisor/GRU_2/zeros:output:0)Supervisor/GRU_2/strided_slice_1:output:0HSupervisor/GRU_2/TensorArrayUnstack/TensorListFromTensor:output_handle:04supervisor_gru_2_gru_cell_55_readvariableop_resource;supervisor_gru_2_gru_cell_55_matmul_readvariableop_resource=supervisor_gru_2_gru_cell_55_matmul_1_readvariableop_resource*
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
#Supervisor_GRU_2_while_body_1124856*/
cond'R%
#Supervisor_GRU_2_while_cond_1124855*8
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
@
Ж
while_body_1126516
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_55_readvariableop_resource_06
2while_gru_cell_55_matmul_readvariableop_resource_08
4while_gru_cell_55_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_55_readvariableop_resource4
0while_gru_cell_55_matmul_readvariableop_resource6
2while_gru_cell_55_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_55/ReadVariableOpReadVariableOp+while_gru_cell_55_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_55/ReadVariableOp 
while/gru_cell_55/unstackUnpack(while/gru_cell_55/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_55/unstackХ
'while/gru_cell_55/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_55_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_55/MatMul/ReadVariableOpг
while/gru_cell_55/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMulЛ
while/gru_cell_55/BiasAddBiasAdd"while/gru_cell_55/MatMul:product:0"while/gru_cell_55/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAddt
while/gru_cell_55/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_55/Const
!while/gru_cell_55/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_55/split/split_dimє
while/gru_cell_55/splitSplit*while/gru_cell_55/split/split_dim:output:0"while/gru_cell_55/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/splitЫ
)while/gru_cell_55/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_55_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_55/MatMul_1/ReadVariableOpМ
while/gru_cell_55/MatMul_1MatMulwhile_placeholder_21while/gru_cell_55/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/MatMul_1С
while/gru_cell_55/BiasAdd_1BiasAdd$while/gru_cell_55/MatMul_1:product:0"while/gru_cell_55/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_55/BiasAdd_1
while/gru_cell_55/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_55/Const_1
#while/gru_cell_55/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_55/split_1/split_dim­
while/gru_cell_55/split_1SplitV$while/gru_cell_55/BiasAdd_1:output:0"while/gru_cell_55/Const_1:output:0,while/gru_cell_55/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_55/split_1Џ
while/gru_cell_55/addAddV2 while/gru_cell_55/split:output:0"while/gru_cell_55/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add
while/gru_cell_55/SigmoidSigmoidwhile/gru_cell_55/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/SigmoidГ
while/gru_cell_55/add_1AddV2 while/gru_cell_55/split:output:1"while/gru_cell_55/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_1
while/gru_cell_55/Sigmoid_1Sigmoidwhile/gru_cell_55/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Sigmoid_1Ќ
while/gru_cell_55/mulMulwhile/gru_cell_55/Sigmoid_1:y:0"while/gru_cell_55/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mulЊ
while/gru_cell_55/add_2AddV2 while/gru_cell_55/split:output:2while/gru_cell_55/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_2
while/gru_cell_55/TanhTanhwhile/gru_cell_55/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/Tanh
while/gru_cell_55/mul_1Mulwhile/gru_cell_55/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_1w
while/gru_cell_55/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_55/sub/xЈ
while/gru_cell_55/subSub while/gru_cell_55/sub/x:output:0while/gru_cell_55/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/subЂ
while/gru_cell_55/mul_2Mulwhile/gru_cell_55/sub:z:0while/gru_cell_55/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/mul_2Ї
while/gru_cell_55/add_3AddV2while/gru_cell_55/mul_1:z:0while/gru_cell_55/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_55/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_55/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_55/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_55_matmul_1_readvariableop_resource4while_gru_cell_55_matmul_1_readvariableop_resource_0"f
0while_gru_cell_55_matmul_readvariableop_resource2while_gru_cell_55_matmul_readvariableop_resource_0"X
)while_gru_cell_55_readvariableop_resource+while_gru_cell_55_readvariableop_resource_0")
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
GRU_1_while_body_1127771(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_05
1gru_1_while_gru_cell_54_readvariableop_resource_0<
8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0>
:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor3
/gru_1_while_gru_cell_54_readvariableop_resource:
6gru_1_while_gru_cell_54_matmul_readvariableop_resource<
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resourceЯ
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
&GRU_1/while/gru_cell_54/ReadVariableOpReadVariableOp1gru_1_while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02(
&GRU_1/while/gru_cell_54/ReadVariableOpВ
GRU_1/while/gru_cell_54/unstackUnpack.GRU_1/while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2!
GRU_1/while/gru_cell_54/unstackз
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02/
-GRU_1/while/gru_cell_54/MatMul/ReadVariableOpы
GRU_1/while/gru_cell_54/MatMulMatMul6GRU_1/while/TensorArrayV2Read/TensorListGetItem:item:05GRU_1/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2 
GRU_1/while/gru_cell_54/MatMulг
GRU_1/while/gru_cell_54/BiasAddBiasAdd(GRU_1/while/gru_cell_54/MatMul:product:0(GRU_1/while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2!
GRU_1/while/gru_cell_54/BiasAdd
GRU_1/while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
GRU_1/while/gru_cell_54/Const
'GRU_1/while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'GRU_1/while/gru_cell_54/split/split_dim
GRU_1/while/gru_cell_54/splitSplit0GRU_1/while/gru_cell_54/split/split_dim:output:0(GRU_1/while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
GRU_1/while/gru_cell_54/splitн
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype021
/GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOpд
 GRU_1/while/gru_cell_54/MatMul_1MatMulgru_1_while_placeholder_27GRU_1/while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2"
 GRU_1/while/gru_cell_54/MatMul_1й
!GRU_1/while/gru_cell_54/BiasAdd_1BiasAdd*GRU_1/while/gru_cell_54/MatMul_1:product:0(GRU_1/while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2#
!GRU_1/while/gru_cell_54/BiasAdd_1
GRU_1/while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2!
GRU_1/while/gru_cell_54/Const_1Ё
)GRU_1/while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)GRU_1/while/gru_cell_54/split_1/split_dimЫ
GRU_1/while/gru_cell_54/split_1SplitV*GRU_1/while/gru_cell_54/BiasAdd_1:output:0(GRU_1/while/gru_cell_54/Const_1:output:02GRU_1/while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2!
GRU_1/while/gru_cell_54/split_1Ч
GRU_1/while/gru_cell_54/addAddV2&GRU_1/while/gru_cell_54/split:output:0(GRU_1/while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add 
GRU_1/while/gru_cell_54/SigmoidSigmoidGRU_1/while/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
GRU_1/while/gru_cell_54/SigmoidЫ
GRU_1/while/gru_cell_54/add_1AddV2&GRU_1/while/gru_cell_54/split:output:1(GRU_1/while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_1І
!GRU_1/while/gru_cell_54/Sigmoid_1Sigmoid!GRU_1/while/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!GRU_1/while/gru_cell_54/Sigmoid_1Ф
GRU_1/while/gru_cell_54/mulMul%GRU_1/while/gru_cell_54/Sigmoid_1:y:0(GRU_1/while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mulТ
GRU_1/while/gru_cell_54/add_2AddV2&GRU_1/while/gru_cell_54/split:output:2GRU_1/while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_2
GRU_1/while/gru_cell_54/TanhTanh!GRU_1/while/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/TanhЗ
GRU_1/while/gru_cell_54/mul_1Mul#GRU_1/while/gru_cell_54/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_1
GRU_1/while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
GRU_1/while/gru_cell_54/sub/xР
GRU_1/while/gru_cell_54/subSub&GRU_1/while/gru_cell_54/sub/x:output:0#GRU_1/while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/subК
GRU_1/while/gru_cell_54/mul_2MulGRU_1/while/gru_cell_54/sub:z:0 GRU_1/while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/mul_2П
GRU_1/while/gru_cell_54/add_3AddV2!GRU_1/while/gru_cell_54/mul_1:z:0!GRU_1/while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/gru_cell_54/add_3§
0GRU_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder!GRU_1/while/gru_cell_54/add_3:z:0*
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
GRU_1/while/Identity_4Identity!GRU_1/while/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
GRU_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"v
8gru_1_while_gru_cell_54_matmul_1_readvariableop_resource:gru_1_while_gru_cell_54_matmul_1_readvariableop_resource_0"r
6gru_1_while_gru_cell_54_matmul_readvariableop_resource8gru_1_while_gru_cell_54_matmul_readvariableop_resource_0"d
/gru_1_while_gru_cell_54_readvariableop_resource1gru_1_while_gru_cell_54_readvariableop_resource_0"5
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
@
Ж
while_body_1128994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_54_readvariableop_resource_06
2while_gru_cell_54_matmul_readvariableop_resource_08
4while_gru_cell_54_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_54_readvariableop_resource4
0while_gru_cell_54_matmul_readvariableop_resource6
2while_gru_cell_54_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_54/ReadVariableOpReadVariableOp+while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_54/ReadVariableOp 
while/gru_cell_54/unstackUnpack(while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_54/unstackХ
'while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_54/MatMul/ReadVariableOpг
while/gru_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMulЛ
while/gru_cell_54/BiasAddBiasAdd"while/gru_cell_54/MatMul:product:0"while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAddt
while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_54/Const
!while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_54/split/split_dimє
while/gru_cell_54/splitSplit*while/gru_cell_54/split/split_dim:output:0"while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/splitЫ
)while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_54/MatMul_1/ReadVariableOpМ
while/gru_cell_54/MatMul_1MatMulwhile_placeholder_21while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMul_1С
while/gru_cell_54/BiasAdd_1BiasAdd$while/gru_cell_54/MatMul_1:product:0"while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAdd_1
while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_54/Const_1
#while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_54/split_1/split_dim­
while/gru_cell_54/split_1SplitV$while/gru_cell_54/BiasAdd_1:output:0"while/gru_cell_54/Const_1:output:0,while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/split_1Џ
while/gru_cell_54/addAddV2 while/gru_cell_54/split:output:0"while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add
while/gru_cell_54/SigmoidSigmoidwhile/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/SigmoidГ
while/gru_cell_54/add_1AddV2 while/gru_cell_54/split:output:1"while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_1
while/gru_cell_54/Sigmoid_1Sigmoidwhile/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Sigmoid_1Ќ
while/gru_cell_54/mulMulwhile/gru_cell_54/Sigmoid_1:y:0"while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mulЊ
while/gru_cell_54/add_2AddV2 while/gru_cell_54/split:output:2while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_2
while/gru_cell_54/TanhTanhwhile/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Tanh
while/gru_cell_54/mul_1Mulwhile/gru_cell_54/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_1w
while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_54/sub/xЈ
while/gru_cell_54/subSub while/gru_cell_54/sub/x:output:0while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/subЂ
while/gru_cell_54/mul_2Mulwhile/gru_cell_54/sub:z:0while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_2Ї
while/gru_cell_54/add_3AddV2while/gru_cell_54/mul_1:z:0while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_54/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_54_matmul_1_readvariableop_resource4while_gru_cell_54_matmul_1_readvariableop_resource_0"f
0while_gru_cell_54_matmul_readvariableop_resource2while_gru_cell_54_matmul_readvariableop_resource_0"X
)while_gru_cell_54_readvariableop_resource+while_gru_cell_54_readvariableop_resource_0")
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
while_cond_1129514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1129514___redundant_placeholder05
1while_while_cond_1129514___redundant_placeholder15
1while_while_cond_1129514___redundant_placeholder25
1while_while_cond_1129514___redundant_placeholder3
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
я
ь
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1130014

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
ч
ъ
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1125045

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
Г
л
,__inference_Supervisor_layer_call_fn_1128405

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
G__inference_Supervisor_layer_call_and_return_conditional_losses_11268922
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
е
Џ
while_cond_1126674
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1126674___redundant_placeholder05
1while_while_cond_1126674___redundant_placeholder15
1while_while_cond_1126674___redundant_placeholder25
1while_while_cond_1126674___redundant_placeholder3
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
GRU_1_while_cond_1128111(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1A
=gru_1_while_gru_1_while_cond_1128111___redundant_placeholder0A
=gru_1_while_gru_1_while_cond_1128111___redundant_placeholder1A
=gru_1_while_gru_1_while_cond_1128111___redundant_placeholder2A
=gru_1_while_gru_1_while_cond_1128111___redundant_placeholder3
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
е
Џ
while_cond_1129673
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1129673___redundant_placeholder05
1while_while_cond_1129673___redundant_placeholder15
1while_while_cond_1129673___redundant_placeholder25
1while_while_cond_1129673___redundant_placeholder3
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
while_body_1128835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_54_readvariableop_resource_06
2while_gru_cell_54_matmul_readvariableop_resource_08
4while_gru_cell_54_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_54_readvariableop_resource4
0while_gru_cell_54_matmul_readvariableop_resource6
2while_gru_cell_54_matmul_1_readvariableop_resourceУ
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
 while/gru_cell_54/ReadVariableOpReadVariableOp+while_gru_cell_54_readvariableop_resource_0*
_output_shapes

:<*
dtype02"
 while/gru_cell_54/ReadVariableOp 
while/gru_cell_54/unstackUnpack(while/gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
while/gru_cell_54/unstackХ
'while/gru_cell_54/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_54_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype02)
'while/gru_cell_54/MatMul/ReadVariableOpг
while/gru_cell_54/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMulЛ
while/gru_cell_54/BiasAddBiasAdd"while/gru_cell_54/MatMul:product:0"while/gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAddt
while/gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_54/Const
!while/gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!while/gru_cell_54/split/split_dimє
while/gru_cell_54/splitSplit*while/gru_cell_54/split/split_dim:output:0"while/gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/splitЫ
)while/gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_54_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype02+
)while/gru_cell_54/MatMul_1/ReadVariableOpМ
while/gru_cell_54/MatMul_1MatMulwhile_placeholder_21while/gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/MatMul_1С
while/gru_cell_54/BiasAdd_1BiasAdd$while/gru_cell_54/MatMul_1:product:0"while/gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
while/gru_cell_54/BiasAdd_1
while/gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
while/gru_cell_54/Const_1
#while/gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#while/gru_cell_54/split_1/split_dim­
while/gru_cell_54/split_1SplitV$while/gru_cell_54/BiasAdd_1:output:0"while/gru_cell_54/Const_1:output:0,while/gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
while/gru_cell_54/split_1Џ
while/gru_cell_54/addAddV2 while/gru_cell_54/split:output:0"while/gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add
while/gru_cell_54/SigmoidSigmoidwhile/gru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/SigmoidГ
while/gru_cell_54/add_1AddV2 while/gru_cell_54/split:output:1"while/gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_1
while/gru_cell_54/Sigmoid_1Sigmoidwhile/gru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Sigmoid_1Ќ
while/gru_cell_54/mulMulwhile/gru_cell_54/Sigmoid_1:y:0"while/gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mulЊ
while/gru_cell_54/add_2AddV2 while/gru_cell_54/split:output:2while/gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_2
while/gru_cell_54/TanhTanhwhile/gru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/Tanh
while/gru_cell_54/mul_1Mulwhile/gru_cell_54/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_1w
while/gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_54/sub/xЈ
while/gru_cell_54/subSub while/gru_cell_54/sub/x:output:0while/gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/subЂ
while/gru_cell_54/mul_2Mulwhile/gru_cell_54/sub:z:0while/gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/mul_2Ї
while/gru_cell_54/add_3AddV2while/gru_cell_54/mul_1:z:0while/gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/gru_cell_54/add_3п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_54/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_54/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
while/Identity_4"j
2while_gru_cell_54_matmul_1_readvariableop_resource4while_gru_cell_54_matmul_1_readvariableop_resource_0"f
0while_gru_cell_54_matmul_readvariableop_resource2while_gru_cell_54_matmul_readvariableop_resource_0"X
)while_gru_cell_54_readvariableop_resource+while_gru_cell_54_readvariableop_resource_0")
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
Г

'__inference_GRU_1_layer_call_fn_1129106
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_11255262
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
р	
Џ
-__inference_gru_cell_55_layer_call_fn_1130028

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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_11256072
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
	
Ё
GRU_2_while_cond_1127542(
$gru_2_while_gru_2_while_loop_counter.
*gru_2_while_gru_2_while_maximum_iterations
gru_2_while_placeholder
gru_2_while_placeholder_1
gru_2_while_placeholder_2*
&gru_2_while_less_gru_2_strided_slice_1A
=gru_2_while_gru_2_while_cond_1127542___redundant_placeholder0A
=gru_2_while_gru_2_while_cond_1127542___redundant_placeholder1A
=gru_2_while_gru_2_while_cond_1127542___redundant_placeholder2A
=gru_2_while_gru_2_while_cond_1127542___redundant_placeholder3
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
ЊX
і
B__inference_GRU_1_layer_call_and_return_conditional_losses_1129084
inputs_0'
#gru_cell_54_readvariableop_resource.
*gru_cell_54_matmul_readvariableop_resource0
,gru_cell_54_matmul_1_readvariableop_resource
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
gru_cell_54/ReadVariableOpReadVariableOp#gru_cell_54_readvariableop_resource*
_output_shapes

:<*
dtype02
gru_cell_54/ReadVariableOp
gru_cell_54/unstackUnpack"gru_cell_54/ReadVariableOp:value:0*
T0* 
_output_shapes
:<:<*	
num2
gru_cell_54/unstackБ
!gru_cell_54/MatMul/ReadVariableOpReadVariableOp*gru_cell_54_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02#
!gru_cell_54/MatMul/ReadVariableOpЉ
gru_cell_54/MatMulMatMulstrided_slice_2:output:0)gru_cell_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMulЃ
gru_cell_54/BiasAddBiasAddgru_cell_54/MatMul:product:0gru_cell_54/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAddh
gru_cell_54/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_54/Const
gru_cell_54/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split/split_dimм
gru_cell_54/splitSplit$gru_cell_54/split/split_dim:output:0gru_cell_54/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/splitЗ
#gru_cell_54/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_54_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype02%
#gru_cell_54/MatMul_1/ReadVariableOpЅ
gru_cell_54/MatMul_1MatMulzeros:output:0+gru_cell_54/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/MatMul_1Љ
gru_cell_54/BiasAdd_1BiasAddgru_cell_54/MatMul_1:product:0gru_cell_54/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ<2
gru_cell_54/BiasAdd_1
gru_cell_54/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      џџџџ2
gru_cell_54/Const_1
gru_cell_54/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_54/split_1/split_dim
gru_cell_54/split_1SplitVgru_cell_54/BiasAdd_1:output:0gru_cell_54/Const_1:output:0&gru_cell_54/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2
gru_cell_54/split_1
gru_cell_54/addAddV2gru_cell_54/split:output:0gru_cell_54/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add|
gru_cell_54/SigmoidSigmoidgru_cell_54/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid
gru_cell_54/add_1AddV2gru_cell_54/split:output:1gru_cell_54/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_1
gru_cell_54/Sigmoid_1Sigmoidgru_cell_54/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Sigmoid_1
gru_cell_54/mulMulgru_cell_54/Sigmoid_1:y:0gru_cell_54/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul
gru_cell_54/add_2AddV2gru_cell_54/split:output:2gru_cell_54/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_2u
gru_cell_54/TanhTanhgru_cell_54/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/Tanh
gru_cell_54/mul_1Mulgru_cell_54/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_1k
gru_cell_54/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_54/sub/x
gru_cell_54/subSubgru_cell_54/sub/x:output:0gru_cell_54/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/sub
gru_cell_54/mul_2Mulgru_cell_54/sub:z:0gru_cell_54/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/mul_2
gru_cell_54/add_3AddV2gru_cell_54/mul_1:z:0gru_cell_54/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gru_cell_54/add_3
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_54_readvariableop_resource*gru_cell_54_matmul_readvariableop_resource,gru_cell_54_matmul_1_readvariableop_resource*
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
while_body_1128994*
condR
while_cond_1128993*8
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
ф
z
%__inference_OUT_layer_call_fn_1129826

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
@__inference_OUT_layer_call_and_return_conditional_losses_11268262
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
Т
р
,__inference_Supervisor_layer_call_fn_1127702
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_11269362
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
і<
к
B__inference_GRU_1_layer_call_and_return_conditional_losses_1125526

inputs
gru_cell_54_1125450
gru_cell_54_1125452
gru_cell_54_1125454
identityЂ#gru_cell_54/StatefulPartitionedCallЂwhileD
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
#gru_cell_54/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_54_1125450gru_cell_54_1125452gru_cell_54_1125454*
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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_11250852%
#gru_cell_54/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_54_1125450gru_cell_54_1125452gru_cell_54_1125454*
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
while_body_1125462*
condR
while_cond_1125461*8
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
IdentityIdentitytranspose_1:y:0$^gru_cell_54/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#gru_cell_54/StatefulPartitionedCall#gru_cell_54/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г

'__inference_GRU_2_layer_call_fn_1129435
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_11259702
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
while_cond_1129174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1129174___redundant_placeholder05
1while_while_cond_1129174___redundant_placeholder15
1while_while_cond_1129174___redundant_placeholder25
1while_while_cond_1129174___redundant_placeholder3
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
'__inference_GRU_1_layer_call_fn_1128755

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
B__inference_GRU_1_layer_call_and_return_conditional_losses_11262592
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
Г

'__inference_GRU_2_layer_call_fn_1129446
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_11260882
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
inputs/0"ИL
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
N_default_save_signature
*O&call_and_return_all_conditional_losses
P__call__"ъ(
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
*Q&call_and_return_all_conditional_losses
R__call__"

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
*S&call_and_return_all_conditional_losses
T__call__"

_tf_keras_rnn_layerь	{"class_name": "GRU", "name": "GRU_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GRU_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [128, 20, 20]}}

_inbound_nodes

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"Щ
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
	variables
&layer_metrics
'metrics
(layer_regularization_losses
trainable_variables
)non_trainable_variables

*layers
regularization_losses
P__call__
N_default_save_signature
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
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
*X&call_and_return_all_conditional_losses
Y__call__"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_54", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_54", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
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
	variables
/layer_metrics
0metrics
1layer_regularization_losses
trainable_variables
2non_trainable_variables

3states

4layers
regularization_losses
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
Ђ

#kernel
$recurrent_kernel
%bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"ч
_tf_keras_layerЭ{"class_name": "GRUCell", "name": "gru_cell_55", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_55", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
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
	variables
9layer_metrics
:metrics
;layer_regularization_losses
trainable_variables
<non_trainable_variables

=states

>layers
regularization_losses
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
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
	variables
?layer_metrics
@metrics
Alayer_regularization_losses
trainable_variables
Bnon_trainable_variables

Clayers
regularization_losses
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
*:(<2GRU_1/gru_cell_54/kernel
4:2<2"GRU_1/gru_cell_54/recurrent_kernel
(:&<2GRU_1/gru_cell_54/bias
*:(<2GRU_2/gru_cell_55/kernel
4:2<2"GRU_2/gru_cell_55/recurrent_kernel
(:&<2GRU_2/gru_cell_55/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
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
+	variables
Dlayer_metrics
Emetrics
Flayer_regularization_losses
,trainable_variables
Gnon_trainable_variables

Hlayers
-regularization_losses
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
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
'
	0"
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
5	variables
Ilayer_metrics
Jmetrics
Klayer_regularization_losses
6trainable_variables
Lnon_trainable_variables

Mlayers
7regularization_losses
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
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
ш2х
"__inference__wrapped_model_1124973О
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
ъ2ч
G__inference_Supervisor_layer_call_and_return_conditional_losses_1127319
G__inference_Supervisor_layer_call_and_return_conditional_losses_1127660
G__inference_Supervisor_layer_call_and_return_conditional_losses_1128043
G__inference_Supervisor_layer_call_and_return_conditional_losses_1128384Р
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
,__inference_Supervisor_layer_call_fn_1128426
,__inference_Supervisor_layer_call_fn_1127702
,__inference_Supervisor_layer_call_fn_1128405
,__inference_Supervisor_layer_call_fn_1127681Р
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
ы2ш
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128925
B__inference_GRU_1_layer_call_and_return_conditional_losses_1129084
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128744
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128585е
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
'__inference_GRU_1_layer_call_fn_1128755
'__inference_GRU_1_layer_call_fn_1129106
'__inference_GRU_1_layer_call_fn_1128766
'__inference_GRU_1_layer_call_fn_1129095е
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129764
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129265
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129424
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129605е
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
'__inference_GRU_2_layer_call_fn_1129446
'__inference_GRU_2_layer_call_fn_1129786
'__inference_GRU_2_layer_call_fn_1129775
'__inference_GRU_2_layer_call_fn_1129435е
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
@__inference_OUT_layer_call_and_return_conditional_losses_1129817Ђ
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
%__inference_OUT_layer_call_fn_1129826Ђ
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
%__inference_signature_wrapper_1126978GRU_1_input
и2е
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1129866
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1129906О
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
-__inference_gru_cell_54_layer_call_fn_1129920
-__inference_gru_cell_54_layer_call_fn_1129934О
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1130014
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1129974О
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
-__inference_gru_cell_55_layer_call_fn_1130028
-__inference_gru_cell_55_layer_call_fn_1130042О
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
 З
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128585q" !?Ђ<
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128744q" !?Ђ<
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_1128925" !OЂL
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
B__inference_GRU_1_layer_call_and_return_conditional_losses_1129084" !OЂL
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
'__inference_GRU_1_layer_call_fn_1128755d" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
'__inference_GRU_1_layer_call_fn_1128766d" !?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЈ
'__inference_GRU_1_layer_call_fn_1129095}" !OЂL
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
'__inference_GRU_1_layer_call_fn_1129106}" !OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџб
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129265%#$OЂL
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129424%#$OЂL
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129605q%#$?Ђ<
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
B__inference_GRU_2_layer_call_and_return_conditional_losses_1129764q%#$?Ђ<
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
'__inference_GRU_2_layer_call_fn_1129435}%#$OЂL
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
'__inference_GRU_2_layer_call_fn_1129446}%#$OЂL
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
'__inference_GRU_2_layer_call_fn_1129775d%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
'__inference_GRU_2_layer_call_fn_1129786d%#$?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЈ
@__inference_OUT_layer_call_and_return_conditional_losses_1129817d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
%__inference_OUT_layer_call_fn_1129826W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџТ
G__inference_Supervisor_layer_call_and_return_conditional_losses_1127319w" !%#$@Ђ=
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_1127660w" !%#$@Ђ=
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_1128043r" !%#$;Ђ8
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
G__inference_Supervisor_layer_call_and_return_conditional_losses_1128384r" !%#$;Ђ8
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
,__inference_Supervisor_layer_call_fn_1127681j" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
,__inference_Supervisor_layer_call_fn_1127702j" !%#$@Ђ=
6Ђ3
)&
GRU_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
,__inference_Supervisor_layer_call_fn_1128405e" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
,__inference_Supervisor_layer_call_fn_1128426e" !%#$;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
"__inference__wrapped_model_1124973s" !%#$8Ђ5
.Ђ+
)&
GRU_1_inputџџџџџџџџџ
Њ "-Њ*
(
OUT!
OUTџџџџџџџџџ
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1129866З" !\ЂY
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
H__inference_gru_cell_54_layer_call_and_return_conditional_losses_1129906З" !\ЂY
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
-__inference_gru_cell_54_layer_call_fn_1129920Љ" !\ЂY
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
-__inference_gru_cell_54_layer_call_fn_1129934Љ" !\ЂY
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1129974З%#$\ЂY
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
H__inference_gru_cell_55_layer_call_and_return_conditional_losses_1130014З%#$\ЂY
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
-__inference_gru_cell_55_layer_call_fn_1130028Љ%#$\ЂY
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
-__inference_gru_cell_55_layer_call_fn_1130042Љ%#$\ЂY
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
%__inference_signature_wrapper_1126978" !%#$GЂD
Ђ 
=Њ:
8
GRU_1_input)&
GRU_1_inputџџџџџџџџџ"-Њ*
(
OUT!
OUTџџџџџџџџџ