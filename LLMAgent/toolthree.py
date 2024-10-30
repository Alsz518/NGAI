import csv
import time
import pandas as pd

from build_simple_road_network import build_simple_road_network
from save_to_csv import save_nodes_to_csv,save_edges_to_csv
from visualize_road_network import visualize_road_network
from create_data_folder import create_data_folder

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class vehiclesCounter:
    def __init__(self, csvFile: str) -> None:
        self.csvFile = csvFile

    @prompts(name='Vehicle Number Counter',
             description="""
             Calculate the number of vehicles from start hour to end hour. 
             The input should be a comma seperated string, representing the start hour and the end hour, which should be integers between 0 and 24. 
             The output is the vehicle numbers during this time preiod.
             """)
    def inference(self, inputs: str) -> str:
        start, end = inputs.replace(' ', '').split(',')
        a, b = int(start), int(end)
        df = pd.read_csv(self.csvFile)
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
        data = df[(df["hour"] >= a) & (df["hour"] < b)]

        return len(data['carid'].unique())


class Evaluation:
    def __init__(self) -> None:
        pass

    @prompts(name="Evaluation",
             description="""
             Evaluation of traffic operation status of a intersection based on vehicle numbers. 
             The input should be a string of a single number, representing the vehicle numbers.
             For example: "40"
             """)
    def inference(self, flow: str) -> str:
        flow = eval(flow)
        if flow < 70:
            return "Normal Operation Status"
        else:
            return "Heavy Traffic Status"


class BuildSimpleRoadNetwork:
    def __init__(self) -> None:
        pass

    @prompts(name='帮我生成一个简单路网',
             description="""
                    基于坐标与生成一个N*M的路网图
                    输入应当包括路网的长和宽，用空格间隔
                    坐标默认为：center_lat=39.125, center_lon=161.567
                    比例尺默认为0.004
                    比如:3,4
                     """)

    def inference(self,flow: str) -> str:
        n,m = flow.replace(' ', ' ').split(',')
        road_network = build_simple_road_network(n,m)
        scale = 0.004

        timestamp = int(time.time())
        create_data_folder('data')
        node_csv_filename = f'data\\nodes_{timestamp}.csv'
        edge_csv_filename = f'data\\edges_{timestamp}.csv'
        image_output_path = f'data\\road_network_{timestamp}.png'

        save_nodes_to_csv(road_network['nodes'], node_csv_filename)
        save_edges_to_csv(road_network['edges'], edge_csv_filename)
        visualize_road_network(road_network['nodes'], road_network['edges'], image_output_path)

        result_string = f"已完成{n}*{m}简单路网绘制\n比例尺为{scale}\n节点文件保存在{node_csv_filename}的地址\n道路文件保存在{edge_csv_filename}的地址\n路网预览图保存在{image_output_path}的地址。"
        return result_string
# 实例化 BuildSimpleRoadNetwork 类
#road_network_builder = BuildSimpleRoadNetwork()

# 获取用户输入
#("请输入路网的长和宽：")
#n, m = input().split()

# 调用 inference 方法
#result = road_network_builder.inference(n, m)
#print(result)