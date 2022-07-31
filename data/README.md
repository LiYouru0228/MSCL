# Upload Data Guidelines:
Upload the dataset for your own task here. The dataset should be organized as a python 'dict', and it contains the following parts:
1. entityid2baseinfo
    + type: dict
    + format: {entityid: one-hot encoding baseinfo list}
2. kg
    + type: dict
    + format: {head entityId: [(tail entityId, relationId),...]}
3. entityid2type
    + type: dict
    + format: {entityId: "B"/"C", etc.}
4. sample_dict
    + type: dict
    + format: {dataset name: numpy.ndarray}
      +  raw demo: <enterpriseID1, enterpriseID2, lable>
