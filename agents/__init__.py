from agents.fql import FQLAgent
from agents.fql_sep import FQLAgent_sep
from agents.fql_sep_v2 import FQLAgent_sep_v2
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.fql_decompose import FQLAgent_dc
from agents.fql_meanfield import FQLAgent_MeanField

agents = dict(
    fql=FQLAgent,
    fql_sep=FQLAgent_sep,
    fql_sep_v2=FQLAgent_sep_v2,
    fql_decompose=FQLAgent_dc,
    fql_meanfield=FQLAgent_MeanField,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
