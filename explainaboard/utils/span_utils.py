from __future__ import annotations


def get_spans_from_bio(bio_seq: list[str]) -> list[tuple[str, int, int]]:
    """
    Takes in a BIO-tagged sequence of tokens, and returns tagged spans.
    :param bio_seq: A sequence of bio-tagged strings
                    such as ['O', 'B-PER', 'I-PER', 'B-ORG', 'O']
    :return: A sequence of spans in format (tag,begin,end),
             such as [('PER',1,3), ('ORG',3,4)]
    """
    default = 'O'
    # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    spans = []
    span_type, span_start = None, -1
    for i, tok in enumerate(bio_seq):
        # End of a span 1
        if tok == default and span_type is not None:
            # Add a span.
            span = (span_type, span_start, i)
            spans.append(span)
            span_type, span_start = None, -1

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
        span = (span_type, span_start, len(bio_seq))
        spans.append(span)

    return spans
