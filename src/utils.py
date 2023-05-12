import copy

###################################################
# Preprocessing tools:
###################################################

def lowercaser_mentions(dd_mentions):
    dd_lowercasedMentions = copy.deepcopy(dd_mentions)
    for id in dd_lowercasedMentions.keys():
        dd_lowercasedMentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_lowercasedMentions


def lowercaser_ref(dd_ref):
    dd_lowercasedRef = copy.deepcopy(dd_ref)
    for cui in dd_ref.keys():
        dd_lowercasedRef[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_lowercasedRef[cui]["tags"] = l_lowercasedTags
    return dd_lowercasedRef


###################################################
# An accuracy function:
###################################################

def accuracy(dd_pred, dd_resp):
    totalScore = 0.0

    for id in dd_resp.keys():
        score = 0.0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]
        if len(l_cuiPred) > 0:  # If there is at least one prediction
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    score += 1
            score = score / max(len(l_cuiResp), len(l_cuiPred))  # multi-norm and too many pred

        totalScore += score  # Must be incremented even if no prediction

    totalScore = totalScore / len(dd_resp.keys())

    return totalScore
