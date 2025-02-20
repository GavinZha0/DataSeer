import json
import networkx as nx
import ray
from sqlalchemy.ext.asyncio import AsyncSession
from apps.datamgr import crud as dm_crud
from apps.ml import crud as ml_crud, model as ml_model
from core.crud import RET
from utils.ray.ray_workflow import RayWorkflow


async def run_workflow(flow_id: int, db: AsyncSession, user: dict):
    # get algo, dataset and datasource info from db
    flow_info = await ml_crud.FlowDal(db).get_data(flow_id, v_ret=RET.SCHEMA, v_where=[ml_model.Workflow.org_id == user.oid])
    json_flow = flow_info.workflow

    # initialize RayPipeline and run training
    ray_flow = RayWorkflow.remote(flow_info)
    ray_tasks = []
    node_to_task = {}

    # build networkX graph
    nx_graph = nx.DiGraph()
    for node in json_flow['nodes']:
        nx_graph.add_node(node['id'], data=node['data'])
    for edge in json_flow['edges']:
        nx_graph.add_edge(edge['source']['cell'], edge['target']['cell'])

    # recursively traverse the graph from root nodes by BFS
    # roots = [node for node, in_degree in nx_graph.in_degree() if in_degree == 0]
    # bfs_nodes = list(nx.bfs_tree(nx_graph, source=roots[0]))

    # recursively traverse the graph from root nodes
    for node in nx.topological_sort(nx_graph):
        task_id = None
        node_data = nx_graph.nodes[node]['data']
        # find dependency nodes
        dep_nodes = list(nx_graph.predecessors(node))
        match node_data['type']:
            case 'source':
                # data loading
                dataset_id = node_data['data']['id']
                dataset_info = await ml_crud.DatasetDal(db).get_data(dataset_id, v_ret=RET.SCHEMA)
                source_info = await dm_crud.DatasourceDal(db).get_data(dataset_info.sourceId, v_ret=RET.SCHEMA)
                task_id = ray_flow.load_data.remote(node_data["kind"], source_info, dataset_info)
            case 'proc':
                # data pre-processing
                task_id = ray_flow.transform.remote(node_data["kind"], node_data['data'], node_to_task[dep_nodes[0]])
            case 'fe':
                # feature engineering
                task_id = ray_flow.feature_eng.remote(node_data["kind"], node_data['data'], node_to_task[dep_nodes[0]])
            case 'ml':
                # machine learning execution
                task_id = ray_flow.ml_fit.remote(node_data["kind"], node_data['data'], node_to_task[dep_nodes[0]])
        if task_id:
            # add task to list
            ray_tasks.append(task_id)
            node_to_task[node] = task_id

    # wait for all tasks to complete
    results = ray.get(ray_tasks)
    # get model evaluation result
    model_eval = results[-1]
    return model_eval.to_dict(orient='records')

