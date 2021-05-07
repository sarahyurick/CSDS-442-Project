import warnings
from causalnex.structure import StructureModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from causalnex.structure.notears import from_pandas

# silence warnings
warnings.filterwarnings("ignore")

sm = StructureModel()

"""
sm.add_edges_from([
    ('health', 'absences'),
    ('health', 'G1')
])
print(sm.edges)
"""

data = pd.read_csv('student-por.csv', delimiter=',')
# print(data.columns)
drop_col = ['school', 'age', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime',
            'nursery']
data = data.drop(columns=drop_col)

# handle non-numeric features
struct_data = data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
# print(non_numeric_columns)
le = LabelEncoder()
for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])

# tabu_edges – list of edges(from, to) not to be included in the graph.
tabu_edges = [('sex', 'freetime'), ('sex', 'health'), ('sex', 'absences')]
# tabu_parent_nodes – list of nodes banned from being a parent of any other nodes.
tabu_parent_nodes = []
# tabu_child_nodes – list of nodes banned from being a child of any other nodes.
tabu_child_nodes = ['sex', 'address']

# run NOTEARS algorithm
sm = from_pandas(struct_data, tabu_edges=tabu_edges,
                 tabu_parent_nodes=tabu_parent_nodes,
                 tabu_child_nodes=tabu_child_nodes)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("NO. POSSIBLE EDGES:", len(sm.edges))
for threshold in thresholds:
    temp = sm.copy()
    temp.remove_edges_below_threshold(threshold)

    print("**************************************************")
    print("THRESHOLD:", threshold)
    print("NO. EDGES:", len(temp.edges))
    print(temp.edges)
    # sm = sm.get_largest_subgraph()
    print("**************************************************")
