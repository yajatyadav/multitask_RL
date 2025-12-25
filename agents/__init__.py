# from agents.acfql import ACFQLAgent
# from agents.acrlpd import ACRLPDAgent

from agents.acifql import ACIFQLAgent
from agents.acbcflowactor import ACBCFlowActorAgent
agents = dict(
    acifql=ACIFQLAgent,
    acbcflowactor=ACBCFlowActorAgent,
)
