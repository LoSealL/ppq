"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Default socket definition for common onnx operators.
"""

# pylint: disable=invalid-name

from mppq.ir.base.opdef import DEFAULT_SOCKET_TABLE, Operation, OpSocket, VLink
from mppq.quant import TargetPrecision


def Reshape(op: Operation) -> OpSocket:
    """
    From Opset 5 - 13:

    Inputs
        data (differentiable) : T
            An input tensor.

        shape (non-differentiable) : tensor(int64)
            Specified shape for output.

    Outputs
        reshaped (differentiable) : T
            Reshaped data.
    """

    return OpSocket(
        op=op,
        in_plat=[TargetPrecision.UNSPECIFIED, TargetPrecision.SOI],
        links=[VLink(in_idx=0, out_idx=0)],
    )


def Pad(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs (2 - 3)
        data (differentiable) : T
            Input tensor.

        pads (non-differentiable) : tensor(int64)
            Tensor of integers indicating the number of padding elements to add or remove
            (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels.
            `pads` should be a 1D tensor of shape [2 * input_rank].
            `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
            where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end,
            the number of pad values added at the end of axis `i`.

        constant_value (optional, non-differentiable) : T
            (Optional) A scalar value to be used if the mode chosen is `constant`
            (by default it is 0, empty string or False).

    Outputs
        output (differentiable) : T
            Tensor after padding.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Gather(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q.
            All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of rank q + (r - 1).
    """

    return OpSocket(
        op=op,
        in_plat=[TargetPrecision.UNSPECIFIED, TargetPrecision.FP32],
        links=[VLink(in_idx=0, out_idx=0)],
    )


def Resize(op: Operation) -> OpSocket:
    """
    From Opset 11 - 13:

    Inputs (1 - 4)
        X (differentiable) : T1
            N-D tensor

        roi (optional, non-differentiable) : T2
            1-D tensor given as [start1, ..., startN, end1, ..., endN],
            where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image.
            It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"

        scales (optional, non-differentiable) : tensor(float)
            The scale array along each dimension. It takes value greater than 0.
            If it's less than 1, it's sampling down, otherwise, it's upsampling.
            The number of elements of 'scales' should be the same as the rank of input 'X'.
            One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified.
            If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.

        sizes (optional, non-differentiable) : tensor(int64)
            The size of the output tensor. The number of elements of 'sizes' should be the same as the rank of input 'X'.
            Only one of 'scales' and 'sizes' can be specified.

    Outputs
        Y (differentiable) : T1
            N-D tensor after resizing
    """

    in_plat = [
        TargetPrecision.UNSPECIFIED,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
    ]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Split(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs (1 - 2)
        input (differentiable) : T
            The tensor to split

        split (optional, non-differentiable) : tensor(int64)
            Optional length of each output. Values should be >= 0.Sum of the values
            must be equal to the dim value at 'axis' specified.

    Outputs (1 - ∞)
        outputs (variadic, differentiable) : T
            One or more outputs forming list of tensors after splitting
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Topk(op: Operation) -> OpSocket:
    """
    From Opset 1 - 11:

    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]

        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve

    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing top K values from the input tensor

        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing the corresponding input tensor indices for the top K values.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI]
    out_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32]
    return OpSocket(
        op=op,
        in_plat=in_plat[: op.num_of_input],
        out_plat=out_plat,
        links=[VLink(in_idx=0, out_idx=0)],
    )


def Tile(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs
        input (differentiable) : T
            Input tensor of any shape.

    repeats (non-differentiable) : T1
        1D int64 tensor of the same length as input's dimension number,
        includes numbers of repeated copies along input's dimensions.

    Outputs
        output (differentiable) : T
            Output tensor of the same dimensions and type as tensor input.
            output_dim[i] = input_dim[i] * repeats[i]
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Expand(op: Operation) -> OpSocket:
    """
    From Opset 8 - 13:

    Inputs
        input (differentiable) : T

    Input tensor
        shape (non-differentiable) : tensor(int64)
            A 1-D tensor indicates the shape you want to expand to, following the broadcast rule

    Outputs
        output (differentiable) : T
            Output tensor
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def RoiAlign(op: Operation) -> OpSocket:
    """
    From Opset 10 - 16:

    Inputs
        X : T1
            Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W),
            where N is the batch size, C is the number of channels,
            and H and W are the height and the width of the data.

        rois : T1
            RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4)
            given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image.
            Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.

        batch_indices : T2
            1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.

    Outputs
        Y : T1
            RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width).
            The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def GridSampler(op: Operation) -> OpSocket:
    """
    From MMCV
    """
    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Clip(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs (1 - 3)
        input (differentiable) : T
            Input tensor whose elements to be clipped

        min (optional, non-differentiable) : T
            Minimum value, under which element is replaced by min.
            It must be a scalar(tensor of empty shape).

        max (optional, non-differentiable) : T
            Maximum value, above which element is replaced by max.
            It must be a scalar(tensor of empty shape).

    Outputs
        output (differentiable) : T
            Output tensor with clipped input elements
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32, TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def ConstantOfShape(op: Operation) -> OpSocket:
    """
    From Opset 9 - 9:

    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor.
            If empty tensor is given, the output would be a scalar. All values must be >= 0.

    Outputs
        output : T2
            Output tensor of shape specified by 'input'.If attribute 'value' is specified,
            the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified,
            the value in the output defaults to 0, and the datatype defaults to float32.
    """

    in_plat = [TargetPrecision.SOI]
    return OpSocket(op=op, in_plat=in_plat[: op.num_of_input], links=[])


def GatherElements(op: Operation) -> OpSocket:
    """
    From Opset 11 - 13:

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, with the same rank r as the input.
            All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of the same shape as indices.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def GatherND(op: Operation) -> OpSocket:
    """
    From Opset 11 - 13:

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def NonMaxSuppression(op: Operation) -> OpSocket:
    """
    From Opset 10 - 13:

    Inputs (2 - 5)
        boxes : tensor(float)
            An input tensor with shape [num_batches, spatial_dimension, 4].
            The single box data format is indicated by center_point_box.

        scores : tensor(float)
            An input tensor with shape [num_batches, num_classes, spatial_dimension]

        max_output_boxes_per_class (optional) : tensor(int64)
            Integer representing the maximum number of boxes to be selected per batch per class.
            It is a scalar. Default to 0, which means no output.

        iou_threshold (optional) : tensor(float)
            Float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
            It is scalar. Value range [0, 1]. Default to 0.

        score_threshold (optional) : tensor(float)
            Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
            selected indices from the boxes tensor. [num_selected_indices, 3],
            the selected index format is [batch_index, class_index, box_index].
    """

    in_plat = [
        TargetPrecision.UNSPECIFIED,
        TargetPrecision.FP32,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
    ]
    return OpSocket(op=op, in_plat=in_plat[: op.num_of_input], links=[])


def NonZero(op: Operation) -> OpSocket:
    """
    From Opset 9 - 13:

    Inputs
        X (non-differentiable) : T
            input

    Outputs
        Y (non-differentiable) : tensor(int64)
            output
    """

    in_plat = [TargetPrecision.UNSPECIFIED]
    out_plat = [TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], out_plat=out_plat, links=[]
    )


def Range(op: Operation) -> OpSocket:
    """
    From Opset 11 - 13:

    Inputs
        start : T
            Scalar. First entry for the range of output values.

        limit : T
            Scalar. Exclusive upper limit for the range of output values.

        delta : T
            Scalar. Value to step by.

    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.
    """

    in_plat = [TargetPrecision.SOI, TargetPrecision.SOI, TargetPrecision.SOI]
    return OpSocket(op=op, in_plat=in_plat[: op.num_of_input], links=[])


def ScatterElements(op: Operation) -> OpSocket:
    """
    From Opset 11 - 16:

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input).
            All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """

    in_plat = [
        TargetPrecision.UNSPECIFIED,
        TargetPrecision.FP32,
        TargetPrecision.UNSPECIFIED,
    ]
    return OpSocket(
        op=op,
        in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)],
    )


def ScatterND(op: Operation) -> OpSocket:
    """
    From Opset 11 - 16:

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1.

        updates (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1.
    """

    in_plat = [
        TargetPrecision.UNSPECIFIED,
        TargetPrecision.FP32,
        TargetPrecision.UNSPECIFIED,
    ]
    return OpSocket(
        op=op,
        in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)],
    )


def Shape(op: Operation) -> OpSocket:
    """
    From Opset 1 - 16:

    Inputs
        data (non-differentiable) : T
            An input tensor.

    Outputs
        shape (non-differentiable) : T1
            Shape of the input tensor
    """

    in_plat = [TargetPrecision.UNSPECIFIED]
    out_plat = [TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], out_plat=out_plat, links=[]
    )


def Slice(op: Operation) -> OpSocket:
    """
    From Opset 1 - 20:

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.
    """

    in_plat = [
        TargetPrecision.UNSPECIFIED,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
        TargetPrecision.SOI,
    ]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Squeeze(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs (1 - 2)
        data (differentiable) : T
            Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to squeeze.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(data).

    Outputs
        squeezed (differentiable) : T
            Reshaped tensor with same data as input.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Unsqueeze(op: Operation) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs
        data (differentiable) : T
            Original tensor

        axes (non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to be inserted.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(expanded).

    Outputs
        expanded (differentiable) : T
            Reshaped tensor with same data as input.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Where(op: Operation) -> OpSocket:
    """
    From Opset 9 - 16:

    Inputs
        condition (non-differentiable) : B
            When True (nonzero), yield X, otherwise yield Y

        X (differentiable) : T
            values selected at indices where condition is True

        Y (differentiable) : T
            values selected at indices where condition is False

    Outputs
        output (differentiable) : T
            Tensor of shape equal to the broadcasted shape of condition, X, and Y.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.FP32, TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], links=[VLink(in_idx=0, out_idx=0)]
    )


def Logical(op: Operation) -> OpSocket:
    """
    From Opset 1 - 16:

    Inputs
        A (non-differentiable) : T
            First input operand for the logical operator.

        B (non-differentiable) : T
            Second input operand for the logical operator.

    Outputs
        C (non-differentiable) : T1
            Result tensor.
    """

    in_plat = [TargetPrecision.UNSPECIFIED, TargetPrecision.UNSPECIFIED]
    out_plat = [TargetPrecision.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], out_plat=out_plat, links=[]
    )


def Onehot(op: Operation) -> OpSocket:
    """
    Inputs
        indices (non-differentiable) : T1

        depth (non-differentiable) : T2

        values (non-differentiable) : T3

    Outputs
        output (non-differentiable) : T3

    Args:
        op (Operation): _description_

    Returns:
        OpSocket: _description_
    """

    in_plat = [TargetPrecision.SOI, TargetPrecision.SOI, TargetPrecision.SOI]
    out_plat = [TargetPrecision.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], out_plat=out_plat, links=[]
    )


DEFAULT_SOCKET_TABLE.update(
    Clip=Clip,
    ConstantOfShape=ConstantOfShape,
    Equal=Logical,
    Expand=Expand,
    Gather=Gather,
    GatherElements=GatherElements,
    GatherND=GatherND,
    grid_sampler=GridSampler,
    Greater=Logical,
    Less=Logical,
    NonMaxSuppression=NonMaxSuppression,
    NonZero=NonZero,
    Pad=Pad,
    Range=Range,
    Reshape=Reshape,
    Resize=Resize,
    ScatterElements=ScatterElements,
    ScatterND=ScatterND,
    Shape=Shape,
    Slice=Slice,
    Split=Split,
    Squeeze=Squeeze,
    Tile=Tile,
    TopK=Topk,
    Unsqueeze=Unsqueeze,
    Where=Where,
    RoiAlign=RoiAlign,
    MMCVRoiAlign=RoiAlign,
    OneHot=Onehot,
    GreaterOrEqual=Logical,
    LessOrEqual=Logical,
    Xor=Logical,
    Or=Logical,
    And=Logical,
)
