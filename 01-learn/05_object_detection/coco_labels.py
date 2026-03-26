"""
COCO 数据集 80 类标签
"""

COCO_CLASSES = [
    "background",    # 0
    "person",        # 1  人
    "bicycle",       # 2  自行车
    "car",           # 3  汽车
    "motorcycle",    # 4  摩托车
    "airplane",      # 5  飞机
    "bus",           # 6  公共汽车
    "train",         # 7  火车
    "truck",         # 8  卡车
    "boat",          # 9  船
    "traffic light", # 10 交通灯
    "fire hydrant",  # 11 消防栓
    "stop sign",     # 13 停车标志
    "parking meter", # 14 停车计时器
    "bench",         # 15 长椅
    "bird",          # 16 鸟
    "cat",           # 17 猫
    "dog",           # 18 狗
    "horse",         # 19 马
    "sheep",         # 20 羊
    "cow",           # 21 牛
    "elephant",      # 22 大象
    "bear",          # 23 熊
    "zebra",         # 24 斑马
    "giraffe",       # 25 长颈鹿
    "backpack",      # 27 背包
    "umbrella",      # 28 雨伞
    "handbag",       # 31 手提包
    "tie",           # 32 领带
    "suitcase",      # 33 手提箱
    "frisbee",       # 34 飞盘
    "skis",          # 35 滑雪板
    "snowboard",     # 36 单板滑雪
    "sports ball",   # 37 运动球
    "kite",          # 38 风筝
    "baseball bat",  # 39 棒球棒
    "baseball glove",# 40 棒球手套
    "skateboard",    # 41 滑板
    "surfboard",     # 42 冲浪板
    "tennis racket", # 43 网球拍
    "bottle",        # 44 瓶子
    "wine glass",    # 46 酒杯
    "cup",           # 47 杯子
    "fork",          # 48 叉子
    "knife",         # 49 刀
    "spoon",         # 50 勺子
    "bowl",          # 51 碗
    "banana",        # 52 香蕉
    "apple",         # 53 苹果
    "sandwich",      # 54 三明治
    "orange",        # 55 橙子
    "broccoli",      # 56 西兰花
    "carrot",        # 57 胡萝卜
    "hot dog",       # 58 热狗
    "pizza",         # 59 披萨
    "donut",         # 60 甜甜圈
    "cake",          # 61 蛋糕
    "chair",         # 62 椅子
    "couch",         # 63 沙发
    "potted plant",  # 64 盆栽植物
    "bed",           # 65 床
    "dining table",  # 67 餐桌
    "toilet",        # 70 马桶
    "tv",            # 72 电视
    "laptop",        # 73 笔记本电脑
    "mouse",         # 74 鼠标
    "remote",        # 75 遥控器
    "keyboard",      # 76 键盘
    "cell phone",    # 77 手机
    "microwave",     # 78 微波炉
    "oven",          # 79 烤箱
    "toaster",       # 80 烤面包机
    "sink",          # 81 水槽
    "refrigerator",  # 82 冰箱
    "book",          # 84 书
    "clock",         # 85 时钟
    "vase",          # 86 花瓶
    "scissors",      # 87 剪刀
    "teddy bear",    # 88 泰迪熊
    "hair drier",    # 89 吹风机
    "toothbrush",    # 90 牙刷
]

# 中文标签映射
COCO_CLASSES_CN = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公共汽车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "交通灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车标志",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "手提箱",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
}


def get_label(class_name, use_chinese=False):
    """获取标签（支持中英文）"""
    if use_chinese and class_name in COCO_CLASSES_CN:
        return COCO_CLASSES_CN[class_name]
    return class_name