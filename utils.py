import argparse


def parseargs(arglist):
    """ This is my version of an argument parser.
        Parameters:
            arglist: the command line list of args
        Returns:
            the result of parsing with the system parser
    """
    parser = argparse.ArgumentParser()

    for onearg in arglist:
        if len(onearg) == 5:
            parser.add_argument(onearg[0], onearg[1], help=onearg[2], type=onearg[3], default=onearg[4])
        else:
            parser.add_argument(onearg[0], onearg[1], help=onearg[2], type=onearg[3])

    args = parser.parse_args()

    return args

def last_only_selector(context_tuple, split):
    """
    The future_context is now wrapped in a list. Thus, len(context_tuple.future_context) == 1 always true.
    """
    convo = context_tuple.current_utterance.get_conversation()
    convo_length = len(convo.get_chronological_utterance_list())

    matches_split = (context_tuple.current_utterance.get_conversation().meta["split"] == split)
    # before_final = (len(context_tuple.future_context) == 1)
    before_final = (len(context_tuple.context) == convo_length-1)
    return (matches_split and before_final)

def full_selector(context_tuple, split):
    """
    The selector for transform does not have access to future context.
    """
    convo = context_tuple.current_utterance.get_conversation()
    convo_length = len(convo.get_chronological_utterance_list())
    matches_split = (context_tuple.current_utterance.get_conversation().meta["split"] == split)
    is_end = (len(context_tuple.context) == convo_length)
    return (matches_split and not is_end)

def new_delta_train_selector(context_tuple, split):
    """
    The future_context is now wrapped in a list. Thus, len(context_tuple.future_context) == 1 always true.
    """
    convo = context_tuple.current_utterance.get_conversation()
    label = convo.meta["has_delta"]
    convo_length = len(convo.get_chronological_utterance_list())

    matches_split = (context_tuple.current_utterance.get_conversation().meta["split"] == split)
    # before_final = (len(context_tuple.future_context) == 1)
    if label:
        before_final = (len(context_tuple.context) == convo_length-1)
    else:
        if convo_length <= 5:
            before_final = (len(context_tuple.context) == convo_length-1)
        else:
            before_final = (len(context_tuple.context) == convo_length-2)

    return (matches_split and before_final)

def old_delta_train_selector(context_tuple, split):
    """
    Rewrite this for OP signals
    """
    convo = context_tuple.current_utterance.get_conversation()
    label = convo.meta["has_delta"]
    convo_length = len(convo.get_chronological_utterance_list())

    matches_split = (context_tuple.current_utterance.get_conversation().meta["split"] == split)
    # before_final = (len(context_tuple.future_context) == 1)
    if label:
        before_final = (len(context_tuple.context) == convo_length-1)
    else:
        if convo_length <= 5:
            before_final = (len(context_tuple.context) == convo_length-1)
        else:
            before_final = (len(context_tuple.context) == convo_length-2)

    return (matches_split and before_final)