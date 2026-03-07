import streamlit as st
import requests

# 1. 画面のタイトルと説明
st.title("🚨 顧客解約予測AIシステム")
st.write("顧客データを入力すると、AIが解約リスクを瞬時に予測します。")

# 2. ユーザーが操作する入力フォーム（今回は代表的な3つだけ！）
st.subheader("顧客データの入力")
tenure = st.slider("契約期間（月）", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("月額料金（ドル）", value=85.50)
contract = st.selectbox("契約形態", ["Month-to-month", "One year", "Two year"])

# 3. 予測ボタンが押された時の処理
if st.button("AIで解約リスクを予測する"):
    
    # APIに送信するデータ（画面の入力値 ＋ 残りの項目はダミー値で固定）
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": tenure,                 # 画面のスライダーの値が入る！
        "InternetService": "Fiber optic",
        "Contract": contract,             # 画面の選択肢が入る！
        "MonthlyCharges": monthly_charges, # 画面の入力値が入る！
        "TotalCharges": 1026.00,
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Num_Additional_Services": 2
    }

    # FastAPIの推論エンドポイントにデータを送信するURL
    api_url = "http://localhost:8000/predict"
    
    try:
        # POSTリクエストを送信
        response = requests.post(api_url, json=payload)
        
        # 成功（200 OK）した場合
        if response.status_code == 200:
            result = response.json()
            
            # 結果を画面にカッコよく表示
            st.subheader("📊 AIの予測結果")
            if result["prediction"] == 1:
                st.error(f"⚠️ {result['message']}")
            else:
                st.success(f"✅ {result['message']}")
                
            # 確率をパーセント表示（小数点第1位まで）
            prob_percent = result["probability"] * 100
            st.info(f"解約確率: {prob_percent:.1f}%")
            
        else:
            st.warning("APIからエラーが返ってきました。データの形式を確認してください。")
            
    except requests.exceptions.ConnectionError:
        st.error("通信エラー: FastAPIサーバーが起動していない可能性があります！")