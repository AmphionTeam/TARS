def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), "All tensors must have the same number of dimensions"

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output
