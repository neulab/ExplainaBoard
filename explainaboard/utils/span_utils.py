from __future__ import annotations


def get_spans_from_bio(seq):
    """
    tags:dic{'per':1,....}
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (span_type, span_start, span_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = 'O'
    # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    spans = []
    span_type, span_start = None, None
    for i, tok in enumerate(seq):
        # End of a span 1
        if tok == default and span_type is not None:
            # Add a span.
            span = (span_type, span_start, i)
            spans.append(span)
            span_type, span_start = None, None

        # End of a span + start of a span!
        elif tok != default:
            tok_span_class, tok_span_type = tok.split('-')
            if span_type is None:
                span_type, span_start = tok_span_type, i
            elif tok_span_type != span_type or tok_span_class == "B":
                span = (span_type, span_start, i)
                spans.append(span)
                span_type, span_start = tok_span_type, i
        else:
            pass
    # end condition
    if span_type is not None:
        span = (span_type, span_start, len(seq))
        spans.append(span)

    return spans
