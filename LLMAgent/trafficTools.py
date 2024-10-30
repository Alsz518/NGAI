import pandas as pd
import osm2gmns as og
import requests
import urllib
from urllib import parse
from urllib import request
import json
from xpinyin import Pinyin
import osmnx as ox
from LLMAgent.getosmTools import RoadNetworkGenerator
from LLMAgent.Computer_Vision import satellite_predict
# from LLMAgent.build_simple_road_network import build_simple_road_network
# from LLMAgent.save_to_csv import save_nodes_to_csv,save_edges_to_csv
# from LLMAgent.visualize_road_network import visualize_road_network
# from LLMAgent.create_data_folder import create_data_folder

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator
        
        
class generatearoadnetworkmap:
    def __init__(self, figfolder: str, filefolder: str) -> None:
        self.figfolder = figfolder
        self.filefolder = filefolder

    @prompts(name="帮我生成一个路网",
             description="""
                基于地名与搜索半径生成一个随机的路网图(搜索半径默认1000m)
                输入应当包括一个地名与搜索半径
                比如：天安门，1000
                """)
    def inference(self, input: str) -> str:
        addr, search_radius = input.replace(' ', '').split(',')
        generator = RoadNetworkGenerator(self.figfolder, self.filefolder)
        csv_file,image_file = generator.generate_road_network_map(addr, search_radius)
        
        return f"CSV 文件和可视化图像已保存。CSV 文件名：{csv_file}，图像文件名：{image_file}"


class satellite_pic_road_extraction:
    def __init__(self, figfolder: str, filefolder: str) -> None:
        self.figfolder = figfolder
        self.filefolder = filefolder

    @prompts(name='Predict Road from Satellite Image',
             description="""
             Recognize a satellite road map image and provide a compressed GMNS file with road node details, along with a visual representation of the recognized road network.
             The input should be a path to the satellite image which needs to be processed.
             The output is a path to the generated zip file containing gmns-style files, and a path to the visual representation of the recognized road network.
             """)

    def inference(self, inputs: str) -> str:
        file_path = inputs # 好像无需解析
        GMNS_path, best_img_name = satellite_predict.predict_road_from_satellite_image(file_path, self.figfolder, self.filefolder)
        return f'The path to the generated zipped gmns file is: `{GMNS_path}`. The path to the visual representation of the recognized road network is: `{best_img_name}`.'


# 还没整理好
# class BuildSimpleRoadNetwork:
#     def __init__(self, figfolder: str) -> None:
        self.figfolder = figfolder

#     @prompts(name='帮我生成一个简单路网',
#              description="""
#                     基于坐标与生成一个N*M的路网图
#                     输入应当包括路网的长和宽，用空格间隔
#                     坐标默认为：center_lat=39.125, center_lon=161.567
#                     比例尺默认为0.004
#                     比如:3,4
#                      """)

#     def inference(self,flow: str) -> str:
#         n,m = flow.replace(' ', ' ').split(',')
#         road_network = build_simple_road_network(n,m)
#         scale = 0.004

#         timestamp = int(time.time())
#         create_data_folder('data')
#         node_csv_filename = f'data\\nodes_{timestamp}.csv'
#         edge_csv_filename = f'data\\edges_{timestamp}.csv'
#         image_output_path = f'data\\road_network_{timestamp}.png'

#         save_nodes_to_csv(road_network['nodes'], node_csv_filename)
#         save_edges_to_csv(road_network['edges'], edge_csv_filename)
#         visualize_road_network(road_network['nodes'], road_network['edges'], image_output_path)

#         result_string = f"已完成{n}*{m}简单路网绘制\n比例尺为{scale}\n节点文件保存在{node_csv_filename}的地址\n道路文件保存在{edge_csv_filename}的地址\n路网预览图保存在{image_output_path}的地址。"
#         return result_string