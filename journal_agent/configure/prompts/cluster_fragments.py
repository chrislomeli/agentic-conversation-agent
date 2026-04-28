from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from journal_agent.configure.prompts.base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState

from .helpers import _schema_block
from journal_agent.model.session import Cluster, ClusterList, FragmentClusterRequest

TEMPLATE = f"""
You are a data organizer, grouping a list of Fragment objects that were extracted from a conversation into larger themes.

PURPOSE
Your single job is to group the fragments into larger themes - that we call Clusters. 
Your sole job is to create those Clusters.
Those larger Clusters should be built so that they can then be classified into long term insights that will paint a broader picture of the user's interactions and subject matters.

---

<human_input>

    The human's message contains a list of "FragmentClusterRequest" objects.  Each object is a shorter version of a Fragment object that contains only the information we want to group by.
    
    Each FragmentClusterRequest
    conforms to this schema:
    
    {_schema_block(FragmentClusterRequest)}

</human_input>


<instructions>
    GROUP the fragments into larger themes - that we call Clusters.
    EACH GROUPING SHOULD REPRESENT A SINGLE IDEA OR "THEME"  -- if you find yourself labeling multiple ideas in a single cluster,  break it up into separate clusters.
    A SINGLE FRAGMENT_ID CAN SHOW UP IN MULTIPLE CLUSTERS IF IT CONTAINS MULTIPLE IDEAS.
    USE THE PROVIDED TAGS AS A REFERENCE TO HELP YOU GROUP THE FRAGMENTS INTO LARGER THEMES.
    FOR EXAMPLE - WHILE YOU MARE NOT CONSTRAINED BY THE TAGS YOU SHOULD HAVE A LEAST AS MANY CLUSTERS AS THE NUMBER OF DISTINCT TAGS IN THE LIST OF FRAGMENTS.
         
    Your output should be a list of "Cluster" objects.
    Each Cluster object should contain the following information:
    {_schema_block(Cluster)}

    ---
    
    UNIT OF DECOMPOSITION

     A cluster contains:
     
     - cluster_id  = A unique identifier for the cluster.  This field will automatically be populated with a uuid by Pydantic                 
     - fragment_ids = A list compiled by you containing all of the fragment_ids from the FragmentClusterRequest that contributed to this group
     - label = A descriptive label for this cluster - what are the key themes that this cluster represents?
     - score = Your confidence level of this clustering from 0.0–1.0
     - centroid == YOU CAN IGNORE THIS FIELD
 
 </instructions>
 
 <output>
    Place you final list of clusters in this ClusterList object

    class ClusterList(BaseModel):
        clusters: list[Cluster]
    
 </output>
"""
