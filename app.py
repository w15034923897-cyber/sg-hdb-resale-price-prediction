import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 设置页面标题
st.set_page_config(page_title="新加坡转售房价预测系统", layout="wide")

st.title("🏠 新加坡转售房价在线预测系统 (XGBoost版)")
st.markdown("通过左侧面板调整房屋参数，右侧将实时显示预测价格。")

# 1. 加载模型 (使用了缓存以提高速度)
@st.cache_resource
def load_model():
    # 确保该文件与 app.py 在同一目录下
    return joblib.load("xgboost_housing_pipeline.pkl")

model = None
try:
    model = load_model()
    st.sidebar.success("✅ 模型已加载")
except FileNotFoundError:
    st.sidebar.error("❌ 找不到模型文件 xgboost_housing_pipeline.pkl，请确认文件与 app.py 在同一目录。")    
except Exception as e:
    st.sidebar.error(f"❌ 无法加载模型: {e}")
    st.stop()

# 2. 侧边栏：用户输入参数
ALL_TOWNS = sorted([
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
    'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
    'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
    'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
    'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
    'TOA PAYOH', 'WOODLANDS', 'YISHUN'
])
 
ALL_FLAT_TYPES = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
 
ALL_FLAT_MODELS = sorted([
    '2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved',
    'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette',
    'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment',
    'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard',
    'Terrace', 'Type S1', 'Type S2'
])

st.sidebar.header("输入房屋特征参数")

def user_input_features()-> pd.DataFrame:
    # 参数范围与 2025 年 HDB 训练数据的取值范围一致
    floor_area_sqm = st.sidebar.slider("房屋面积 Floor Area (sqm)", 31.0, 195.0, 80.0, help="单位：平方米")
    remaining_lease = st.sidebar.slider("剩余租约 Remaining Lease (年)", 40.0, 95.5, 75.0, step=0.5)
    storey_mid = st.sidebar.slider("楼层中值 Storey Mid", 2.0, 50.0, 8.0, step=3.0, help="HDB 楼层按 3 层一档，如 04 TO 06 的中值为 5")
    
    town = st.sidebar.selectbox("所属城镇 Town", options=ALL_TOWNS, index=ALL_TOWNS.index('TAMPINES'))
    flat_type = st.sidebar.selectbox("房型 Flat Type", options=ALL_FLAT_TYPES, index=2)
    flat_model = st.sidebar.selectbox("户型 Flat Model", options=ALL_FLAT_MODELS)

    data = {
        'floor_area_sqm' : [floor_area_sqm],
        'remaining_lease': [remaining_lease],
        'storey_mid'     : [storey_mid],
        'flat_type'      : [flat_type],
        'town'           : [town],
        'flat_model'     : [flat_model],
    }
        
    return pd.DataFrame(data)

input_df = user_input_features()

# 3. 主界面显示
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 当前输入参数")
    display_df = input_df.rename(columns={
        'floor_area_sqm' : '面积 (sqm)',
        'remaining_lease': '剩余租约 (年)',
        'storey_mid'     : '楼层',
        'flat_type'      : '房型',
        'town'           : '城镇',
        'flat_model'     : '户型',
    })
    st.dataframe(display_df, use_container_width=True)

with col2:
    st.subheader("💰 预测结果")
    if model is not None:
        try:
            prediction = model.predict(input_df)
            final_price = float(prediction[0])
 
            st.metric(
                label="预测转售价格 (SGD)",
                value=f"${final_price:,.0f}"
            )
    
    # 根据价格给出简单评价
            if final_price > 900_000:
                st.warning("⚠️ 这是一套高价房源（>$900K）。")
            elif final_price < 400_000:
                st.info("ℹ️ 这是一套经济型房源（<$400K）。")
            else:
                st.success("✅ 这是一套中等价位房源。")

            # 价格置信参考区间（±10% 粗估）
            st.caption(
                f"参考区间（±10%）：${final_price * 0.9:,.0f}  —  ${final_price * 1.1:,.0f}"
            )
 
        except Exception as e:
            st.error(f"预测失败：{e}")
    else:
        st.warning("模型未加载，无法预测。请检查左侧错误提示。")

# 4.  城镇经纬度地图
TOWN_COORDS = {
    'ANG MO KIO'      : (1.3691, 103.8454),
    'BEDOK'           : (1.3236, 103.9273),
    'BISHAN'          : (1.3526, 103.8352),
    'BUKIT BATOK'     : (1.3490, 103.7495),
    'BUKIT MERAH'     : (1.2819, 103.8239),
    'BUKIT PANJANG'   : (1.3774, 103.7719),
    'BUKIT TIMAH'     : (1.3294, 103.8021),
    'CENTRAL AREA'    : (1.2966, 103.8517),
    'CHOA CHU KANG'   : (1.3840, 103.7470),
    'CLEMENTI'        : (1.3162, 103.7649),
    'GEYLANG'         : (1.3201, 103.8918),
    'HOUGANG'         : (1.3612, 103.8863),
    'JURONG EAST'     : (1.3329, 103.7436),
    'JURONG WEST'     : (1.3404, 103.7090),
    'KALLANG/WHAMPOA' : (1.3100, 103.8651),
    'MARINE PARADE'   : (1.3020, 103.9070),
    'PASIR RIS'       : (1.3721, 103.9474),
    'PUNGGOL'         : (1.3984, 103.9072),
    'QUEENSTOWN'      : (1.2942, 103.7861),
    'SEMBAWANG'       : (1.4491, 103.8185),
    'SENGKANG'        : (1.3868, 103.8914),
    'SERANGOON'       : (1.3554, 103.8679),
    'TAMPINES'        : (1.3496, 103.9568),
    'TOA PAYOH'       : (1.3343, 103.8563),
    'WOODLANDS'       : (1.4382, 103.7890),
    'YISHUN'          : (1.4304, 103.8354),
}
 
st.subheader("📍 所选城镇位置")
selected_town = input_df['town'].iloc[0]
if selected_town in TOWN_COORDS:
    lat, lon = TOWN_COORDS[selected_town]
    map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_df, zoom=13)
else:
    st.info("地图暂无该城镇坐标数据。")
 
# ── 6. 页脚说明 ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 本预测结果基于 2025 年 HDB 转售数据训练的 XGBoost 模型，仅供参考，"
    "实际成交价受个体房源因素影响。"
)