# from agents.acfql import ACFQLAgent
# from agents.acrlpd import ACRLPDAgent

from agents.acifql import ACIFQLAgent
from agents.acbcflowactor import ACBCFlowActorAgent
from agents.trans_classifier import ClassifierBestofNAgent
agents = dict(
    acifql=ACIFQLAgent,
    acbcflowactor=ACBCFlowActorAgent,
    classifier_bestofn=ClassifierBestofNAgent,
)
