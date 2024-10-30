import osm2gmns as og
import requests
import urllib
import json
from xpinyin import Pinyin
import osmnx as ox
import matplotlib.pyplot as plt

class RoadNetworkGenerator:
    def __init__(self, figfolder: str, filefolder: str) -> None:
        self.figfolder = figfolder
        self.filefolder = filefolder

    @staticmethod
    def get_road_network(lat, lon, radius):
        one_mile = radius  # 米
        G = ox.graph_from_point((lat, lon), dist=one_mile, network_type='all_private', simplify=False)
        return G

    @staticmethod
    def output_net_to_csv(net, csv_file):
        og.consolidateComplexIntersections(net, auto_identify=True)
        og.outputNetToCSV(net, csv_file)

    @staticmethod
    def get_geocode(addr):
        key = '97267e642f34dc2df3c890f74c6b8205'
        baseUrl = 'https://restapi.amap.com/v3/geocode/geo?'
        params = {
            'key': key,
            'address': addr
        }
        url = baseUrl + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        content = urllib.request.urlopen(req).read()
        jsonData = json.loads(content)
        lon, lat = '', ''
        if jsonData['status'] == '1':
            try:
                corr = jsonData['geocodes'][0]['location']
                lon, lat = corr.split(',')[0], corr.split(',')[1]
            except:
                lon, lat = '0', '0'
        else:
            print('出错了')
        return (lon, lat)

    @staticmethod
    def get_road_network_data(lat, lon, radius, addr):
        overpass_url = "http://www.overpass-api.de/api/interpreter"

        query = f"""
        [out:xml];
        way
          (around:{radius},{lat},{lon})["highway"];
        (._;>;);
        out meta;
        """

        # 设置请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.60",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # 发送 HTTP POST 请求
        response = requests.post(overpass_url, data=query.encode('utf-8'), headers=headers)

        # 将响应文本保存为 OSM 文件
        if response.status_code == 200:
            with open(addr, "w", encoding="utf-8") as f:
                f.write(response.text)

    @staticmethod
    def generate_road_network_map(self, addr, search_radius):

        # 调用函数获取经纬度
        longitude, latitude = RoadNetworkGenerator.get_geocode(addr)

        # 修改存储路径
        addr_pinyin = Pinyin().get_pinyin(addr, '')
        csv_file = f'{self.filefolder}{addr_pinyin}.csv'

        # 调用函数获取路网数据
        RoadNetworkGenerator.get_road_network_data(latitude, longitude, search_radius, addr_pinyin + '.osm')

        # 转换为 CSV 文件
        net = og.getNetFromFile(addr_pinyin + '.osm')
        RoadNetworkGenerator.output_net_to_csv(net, csv_file)

        # 获取地图数据并可视化
        G = RoadNetworkGenerator.get_road_network(float(latitude), float(longitude), int(search_radius))

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
        ox.plot_graph(ox.project_graph(G), bgcolor='white', node_size=0,edge_color='black',ax=ax)

        # 输出可视化结果
        image_file = f'{self.figfolder}{addr_pinyin}.png'
        fig.savefig(image_file)
        return csv_file,image_file
