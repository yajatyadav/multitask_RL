from agents.iql import IQLAgent
from agents.iql_pi0actor import IQLPi0ActorAgent
from agents.iql_pi0actor_chunked import IQLPi0ActorAgentChunked
agents = dict(
    iql=IQLAgent,
    iql_pi0actor=IQLPi0ActorAgent,
    iql_pi0actor_chunked=IQLPi0ActorAgentChunked,
)