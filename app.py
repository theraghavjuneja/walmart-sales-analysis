import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import io
from PIL import Image
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Retail Sales Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #333;
        margin-top: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-header'>📊 Retail Sales Analysis & Forecasting (Raghav & Shivaansh)</div>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    stores = pd.read_csv('stores.csv')
    features = pd.read_csv('features (1).csv')
    
# Here i am storing unique id per store department to uniquely identify each dept of each store
    train['Store_Dept'] = train['Store'].astype(str) + '_' + train['Dept'].astype(str)
    test['Store_Dept'] = test['Store'].astype(str) + '_' + test['Dept'].astype(str)
# Converting to time-series(Data processing)
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    features['Date'] = pd.to_datetime(features['Date'])
    
# Here i am creating all time based features out of date, (month, year, quarter)  
    train['Month'] = train['Date'].dt.month
    train['Year'] = train['Date'].dt.year
    train['Quarter'] = train['Date'].dt.quarter
    train['WeekOfYear'] = train['Date'].dt.isocalendar().week
# adding the same features for test as ell
    test['Month'] = test['Date'].dt.month
    test['Year'] = test['Date'].dt.year
    test['Quarter'] = test['Date'].dt.quarter
    test['WeekOfYear'] = test['Date'].dt.isocalendar().week
# for feat too 
    features['Month'] = features['Date'].dt.month
    features['Year'] = features['Date'].dt.year


# markdown was new to me, somewhere empty somewhere not, so i just added up all of tje
# this may be related to some uniqye thing, so whenever it was missing I added 0
    
    features['Total_MarkDown'] = features['MarkDown1'].fillna(0) + features['MarkDown2'].fillna(0) + \
                              features['MarkDown3'].fillna(0) + features['MarkDown4'].fillna(0) + \
                              features['MarkDown5'].fillna(0)
    
# Not stores['type] is A,B,C each has different size, so applying one hot encoind
    store_type_dummies = pd.get_dummies(stores['Type'], prefix='Store_Type', drop_first=True)
    stores = pd.concat([stores, store_type_dummies], axis=1)
    
#    now based on merge, merging datasets, (like)
# train merged with Store based on Store (so that for each of them I have store then type then other info)
    train = train.merge(stores, on='Store', how='left')
    # for other info ofcourse i will again merge with features
    train = train.merge(features, on=['Store', 'Date'], how='left')
    # i will aplly same merge on test
    test = test.merge(stores, on='Store', how='left')
    test = test.merge(features, on=['Store', 'Date'], how='left')
    
    
    store_depts = train['Store_Dept'].unique()
    # now i will be adding lag features too
    # like weekly sales lag , and some trends
    # prior 2 week sales and last 4 weeks rolling mean
    for sd in store_depts:

        # to ensure time independent features are of same dept only. Electronics & Baby Clothes shouldnt me mixed
        mask = train['Store_Dept'] == sd
        train.loc[mask, 'Weekly_Sales_Lag1'] = train.loc[mask, 'Weekly_Sales'].shift(1)
        train.loc[mask, 'Weekly_Sales_Lag2'] = train.loc[mask, 'Weekly_Sales'].shift(2)
        train.loc[mask, 'Rolling_Mean'] = train.loc[mask, 'Weekly_Sales'].rolling(window=4).mean()
        train.loc[mask, 'Rolling_Std'] = train.loc[mask, 'Weekly_Sales'].rolling(window=4).std()
    
    
    train = train.sort_values('Date')
    test = test.sort_values('Date')
    features = features.sort_values('Date')
    
   
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    features.fillna(0, inplace=True)
    
    return train, test, stores, features

with st.spinner('Loading data...'):
    train, test, stores, features = load_data()


st.sidebar.title("Navigation")
selected_section = st.sidebar.radio("Go to", ["Data Visualization", "Sales Prediction", "Key Factors Analysis"])


all_stores = sorted(train['Store'].unique())
all_depts = sorted(train['Dept'].unique())


if selected_section == "Data Visualization":
    st.markdown("<div class='sub-header'>Data Visualization</div>", unsafe_allow_html=True)
    
    viz_option = st.selectbox(
        "Select Visualization",
        ["Store Analysis", "Sales Analysis", "Correlation Analysis", "Time Series Analysis"]
    )
    
    if viz_option == "Store Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
        
            fig = px.bar(
                stores['Type'].value_counts().reset_index(),
                x='Type', y='count',
                title='Count of Store Types',
                color='Type',
                labels={'count': 'Count', 'Type': 'Store Type'},
                height=400
            )
            fig.update_layout(xaxis_title='Store Type', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
            
            
            fig = px.box(
                stores, 
                x='Type', 
                y='Size',
                color='Type',
                title='Store Size by Type',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            
            fig = px.histogram(
                stores, 
                x='Size', 
                nbins=20,
                marginal='box',
                title='Distribution of Store Sizes',
                height=400
            )
            fig.update_layout(xaxis_title='Store Size', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
    elif viz_option == "Sales Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
           
            fig = px.histogram(
                train, 
                x='Weekly_Sales',
                nbins=50,
                marginal='box',
                title='Distribution of Weekly Sales',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
          
            dept_sales = train.groupby('Dept')['Weekly_Sales'].mean().reset_index()
            dept_sales = dept_sales.sort_values('Weekly_Sales', ascending=False).head(20)
            
            fig = px.bar(
                dept_sales,
                x='Dept',
                y='Weekly_Sales',
                title='Average Weekly Sales by Top 20 Departments',
                height=500
            )
            fig.update_layout(xaxis_title='Department', yaxis_title='Average Weekly Sales')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
           
            store_sales = train.groupby('Store')['Weekly_Sales'].mean().reset_index()
            store_sales = store_sales.sort_values('Weekly_Sales', ascending=False).head(20)
            
            fig = px.bar(
                store_sales,
                x='Store',
                y='Weekly_Sales',
                title='Average Weekly Sales by Top 20 Stores',
                height=400
            )
            fig.update_layout(xaxis_title='Store', yaxis_title='Average Weekly Sales')
            st.plotly_chart(fig, use_container_width=True)
            
          
            type_sales = train.groupby('Type')['Weekly_Sales'].mean().reset_index()
            
            fig = px.bar(
                type_sales,
                x='Type',
                y='Weekly_Sales',
                color='Type',
                title='Average Weekly Sales by Store Type',
                height=500
            )
            fig.update_layout(xaxis_title='Store Type', yaxis_title='Average Weekly Sales')
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Correlation Analysis":
        st.markdown("<div class='section-header'>Correlation Analysis</div>", unsafe_allow_html=True)

        # Identify numerical columns, excluding unwanted markdowns
        numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if 'Markdown' not in col or col == 'Total_MarkDown']

        # Default columns for selection
        default_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_MarkDown', 'Size']

        selected_cols = st.multiselect(
            "Select features for correlation analysis",
            options=numerical_cols,
            default=default_cols
        )

        if selected_cols:
            # Compute correlation matrix
            corr_matrix = train[selected_cols].corr()

            # Display heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title='Correlation Matrix',
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sample data once for further plots
            sample_size = min(5000, len(train))
            train_sample = train.sample(sample_size, random_state=42)

            # Pairplot only if small number of features selected
            if len(selected_cols) <= 5:
                st.markdown("<div class='section-header'>Pairplot Analysis</div>", unsafe_allow_html=True)

                fig = px.scatter_matrix(
                    train_sample,
                    dimensions=selected_cols,
                    title='Feature Pairplot',
                    opacity=0.5
                )
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)

            # Scatter plots for each feature vs Weekly Sales
            st.markdown("<div class='section-header'>Feature vs Weekly Sales</div>", unsafe_allow_html=True)

            features_to_plot = [col for col in selected_cols if col != 'Weekly_Sales']

            if features_to_plot:
                for feature in features_to_plot:
                    fig = px.scatter(
                        train_sample,
                        x=feature,
                        y='Weekly_Sales',
                        opacity=0.5,
                        title=f'Weekly Sales vs {feature}',
                        trendline='ols'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif viz_option == "Time Series Analysis":
        st.markdown("<div class='section-header'>Time Series Analysis</div>", unsafe_allow_html=True)
        
        
        col1, col2 = st.columns(2)
        with col1:
            selected_store = st.selectbox("Select Store", all_stores)
        with col2:
            filtered_depts = sorted(train[train['Store'] == selected_store]['Dept'].unique())
            selected_dept = st.selectbox("Select Department", filtered_depts)
        
       
        filtered_data = train[(train['Store'] == selected_store) & (train['Dept'] == selected_dept)]
        filtered_data = filtered_data.sort_values('Date')

        
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
        filtered_data['Month'] = filtered_data['Date'].dt.month
        filtered_data['Year'] = filtered_data['Date'].dt.year
        
        
        fig = px.line(
            filtered_data,
            x='Date',
            y='Weekly_Sales',
            title=f'Weekly Sales Over Time for Store {selected_store}, Dept {selected_dept}',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        
        monthly_sales = filtered_data.groupby('Month')['Weekly_Sales'].mean().reset_index()
        fig = px.bar(
            monthly_sales,
            x='Month',
            y='Weekly_Sales',
            title=f'Average Monthly Sales for Store {selected_store}, Dept {selected_dept}',
            color='Month'
        )
        st.plotly_chart(fig, use_container_width=True)
        

        if len(filtered_data['Year'].unique()) > 1:
            yearly_comparison = filtered_data.groupby(['Year', 'Month'])['Weekly_Sales'].mean().reset_index()
            fig = px.line(
                yearly_comparison,
                x='Month',
                y='Weekly_Sales',
                color='Year',
                title=f'Year over Year Sales Comparison for Store {selected_store}, Dept {selected_dept}',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

elif selected_section == "Sales Prediction":
    st.markdown("<div class='sub-header'>Sales Prediction</div>", unsafe_allow_html=True)
    
    
    st.sidebar.markdown("### Forecast Settings")
    
    selected_store = st.sidebar.selectbox("Select Store", all_stores)
    filtered_depts = sorted(train[train['Store'] == selected_store]['Dept'].unique())
    selected_dept = st.sidebar.selectbox("Select Department", filtered_depts)
    
    forecast_periods = st.sidebar.slider("Forecast Periods (Weeks)", 1, 26, 12)
    
   
    model_type = st.sidebar.selectbox(
        "Select Model Type", 
        ["ARIMA", "SARIMA (Seasonal)", "Prophet", "Ensemble"]
    )
    
    
    p = 1
    d = 1
    q = 1
    P = 1
    D = 1
    Q = 1
    m = 52 # seasonal cycle(weeks in a year)
    yearly_seasonality = True
    weekly_seasonality = True
    daily_seasonality = False
    include_holidays = True
    use_arima = True
    use_sarima = True
    use_prophet = True
    ensemble_method = "Mean"
    arima_weight = 0.33
    sarima_weight = 0.33
    prophet_weight = 0.34
    
   
    if model_type == "ARIMA":
        st.sidebar.markdown("### ARIMA Parameters")
        p = st.sidebar.slider("p (AR Order)", 0, 4, 1)
        d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
        q = st.sidebar.slider("q (MA Order)", 0, 4, 1)
    
    elif model_type == "SARIMA (Seasonal)":
        st.sidebar.markdown("### SARIMA Parameters")
        p = st.sidebar.slider("p (AR Order)", 0, 4, 1)
        d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
        q = st.sidebar.slider("q (MA Order)", 0, 4, 1)
        
        st.sidebar.markdown("### Seasonal Parameters")
        P = st.sidebar.slider("P (Seasonal AR)", 0, 2, 1)
        D = st.sidebar.slider("D (Seasonal Differencing)", 0, 1, 1)
        Q = st.sidebar.slider("Q (Seasonal MA)", 0, 2, 1)
        # Fix: Use selectbox instead of slider for fixed options
        m = st.sidebar.selectbox("m (Seasonal Period)", [4, 12, 52], 2)  # Default to 52 (index 2)
    
    elif model_type == "Prophet":
        st.sidebar.markdown("### Prophet Parameters")
        yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", True)
        weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", True)
        daily_seasonality = st.sidebar.checkbox("Daily Seasonality", False)
        include_holidays = st.sidebar.checkbox("Include Holidays", True)
        
    elif model_type == "Ensemble":
        st.sidebar.markdown("### Ensemble Settings")
        use_arima = st.sidebar.checkbox("Include ARIMA", True)
        use_sarima = st.sidebar.checkbox("Include SARIMA", True)
        use_prophet = st.sidebar.checkbox("Include Prophet", True)
        ensemble_method = st.sidebar.selectbox("Ensemble Method", ["Mean", "Median", "Weighted"])
        
        # If ARIMA or SARIMA are selected, allow setting their parameters
        if use_arima or use_sarima:
            st.sidebar.markdown("### Model Parameters")
            p = st.sidebar.slider("p (AR Order)", 0, 4, 1)
            d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
            q = st.sidebar.slider("q (MA Order)", 0, 4, 1)
            
            if use_sarima:
                P = st.sidebar.slider("P (Seasonal AR)", 0, 2, 1)
                D = st.sidebar.slider("D (Seasonal Differencing)", 0, 1, 1)
                Q = st.sidebar.slider("Q (Seasonal MA)", 0, 2, 1)
                m = st.sidebar.selectbox("m (Seasonal Period)", [4, 12, 52], 2)  # Default to 52 (index 2)
        
        if ensemble_method == "Weighted":
            arima_weight = st.sidebar.slider("ARIMA Weight", 0.0, 1.0, 0.3, 0.1) if use_arima else 0
            sarima_weight = st.sidebar.slider("SARIMA Weight", 0.0, 1.0, 0.3, 0.1) if use_sarima else 0
            prophet_weight = st.sidebar.slider("Prophet Weight", 0.0, 1.0, 0.4, 0.1) if use_prophet else 0
            
            # Normalize weights
            total_weight = arima_weight + sarima_weight + prophet_weight
            if total_weight > 0:
                arima_weight /= total_weight
                sarima_weight /= total_weight
                prophet_weight /= total_weight
    
    
    st.sidebar.markdown("### Additional Features")
    include_features = st.sidebar.checkbox("Include External Features", False)
    
    if include_features:
        # Merge with features data
        include_holiday = st.sidebar.checkbox("Holiday Indicators", True)
        include_markdown = st.sidebar.checkbox("Markdown Events", True)
        include_temperature = st.sidebar.checkbox("Temperature", False)
        include_fuel_price = st.sidebar.checkbox("Fuel Price", False)
        include_unemployment = st.sidebar.checkbox("Unemployment Rate", False)
        include_cpi = st.sidebar.checkbox("CPI", False)
        
    
    sales_data = train[(train['Store'] == selected_store) & (train['Dept'] == selected_dept)]
    
    
    sales_data = sales_data.sort_values('Date')
    time_series = sales_data.set_index('Date')['Weekly_Sales']
    
    # Check if we have enough data (for a relibale forecast, although this is always true)
    if len(sales_data) < 10:
        st.warning(f"Not enough data points for Store {selected_store} and Department {selected_dept} to create a reliable forecast.")
    else:
        
        st.markdown("<div class='section-header'>Historical Weekly Sales</div>", unsafe_allow_html=True)
        
        # 3 sections time series, yealry & seasonlit
        tab1, tab2, tab3 = st.tabs(["Time Series", "Yearly Comparison", "Seasonal Decomposition"])
        
        with tab1:
        
            fig = px.line(
                sales_data,
                x='Date',
                y='Weekly_Sales',
                title=f'Historical Weekly Sales for Store {selected_store}, Dept {selected_dept}',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            
            sales_data['Year'] = sales_data['Date'].dt.year
            sales_data['Week'] = sales_data['Date'].dt.isocalendar().week
            
            yearly_fig = px.line(
                sales_data,
                x='Week',
                y='Weekly_Sales',
                color='Year',
                title=f'Yearly Sales Comparison for Store {selected_store}, Dept {selected_dept}',
                markers=True
            )
            yearly_fig.update_xaxes(title="Week of Year")
            st.plotly_chart(yearly_fig, use_container_width=True)
        
        with tab3:
        
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
            
                if len(time_series) >= 52*2:  # Need at least 2 years for yearly seasonality
                    decomposition = seasonal_decompose(time_series, period=52)
                    
                    
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
                        vertical_spacing=0.1
                    )
                    
                    
                    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name="Observed"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name="Trend"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name="Seasonal"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name="Residual"), row=4, col=1)
                    
                    fig.update_layout(height=800, title_text="Seasonal Decomposition of Sales")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 years of data for seasonal decomposition.")
            except Exception as e:
                st.error(f"Error in seasonal decomposition: {e}")
                st.info("Try selecting a store/department with more continuous data.")
        

        exog_data = None
        future_exog = None
        
        if include_features:
            try:
                
                store_features = features[features['Store'] == selected_store].copy()
                
                
                feature_cols = []
                if include_holiday:
                    feature_cols.extend(['IsHoliday'])
                if include_markdown:
                    markdown_cols = [col for col in store_features.columns if 'MarkDown' in col]
                    feature_cols.extend(markdown_cols)
                if include_temperature:
                    feature_cols.extend(['Temperature'])
                if include_fuel_price:
                    feature_cols.extend(['Fuel_Price'])
                if include_unemployment:
                    feature_cols.extend(['Unemployment'])
                if include_cpi:
                    feature_cols.extend(['CPI'])
                
                if feature_cols:
                   
                    store_features['Date'] = pd.to_datetime(store_features['Date'])
                    merged_data = pd.merge(sales_data[['Date', 'Weekly_Sales']], 
                                          store_features[['Date'] + feature_cols],
                                          on='Date', how='left')
                    
                    
                    merged_data[feature_cols] = merged_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
                    
                    
                    exog_data = merged_data[feature_cols]
                    
                    # (simple forecasting by copying last values)
                    last_date = merged_data['Date'].max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), 
                                                periods=forecast_periods, freq='7D')
                    
                    future_exog = pd.DataFrame(index=future_dates)
                    
                    # For each feature, copy the last known values or use seasonal patterns
                    for col in feature_cols:
                        if 'IsHoliday' in col:
                            # For holidays, we can try to match with previous years
                            future_exog[col] = 0  # Default to no holiday
                            
                            # Check for holidays in the same week in previous years
                            for i, future_date in enumerate(future_dates):
                                week_of_year = future_date.isocalendar()[1]
                                holiday_weeks = merged_data[merged_data['IsHoliday'] == 1]['Date'].dt.isocalendar().week
                                if week_of_year in holiday_weeks.values:
                                    future_exog.loc[future_date, col] = 1
                        else:
                            # For other features, use the last known value
                            future_exog[col] = merged_data[col].iloc[-1]
                    
                    st.markdown("<div class='section-header'>External Features</div>", unsafe_allow_html=True)
                    st.write("Selected features:", ", ".join(feature_cols))
            except Exception as e:
                st.error(f"Error preparing external features: {e}")
                exog_data = None
                future_exog = None
        
        # Fit model and forecast
        with st.spinner('Creating forecast...'):
            try:
                forecast_results = {}
                
                # ARIMA Model
                if model_type == "ARIMA" or (model_type == "Ensemble" and use_arima):
                    model_arima = ARIMA(time_series, order=(p, d, q), exog=exog_data)
                    model_fit_arima = model_arima.fit()
                    
                    # Generate forecast
                    arima_forecast = model_fit_arima.forecast(steps=forecast_periods, exog=future_exog)
                    forecast_results['ARIMA'] = arima_forecast
                
                # SARIMA Model
                if model_type == "SARIMA (Seasonal)" or (model_type == "Ensemble" and use_sarima):
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    
                    model_sarima = SARIMAX(
                        time_series, 
                        exog=exog_data,
                        order=(p, d, q), 
                        seasonal_order=(P, D, Q, m)
                    )
                    model_fit_sarima = model_sarima.fit(disp=False)
                    
                    # Generate forecast
                    sarima_forecast = model_fit_sarima.forecast(steps=forecast_periods, exog=future_exog)
                    forecast_results['SARIMA'] = sarima_forecast
                
                # Prophet Model
                if model_type == "Prophet" or (model_type == "Ensemble" and use_prophet):
                    try:
                        from prophet import Prophet
                        
                        # Prepare data for Prophet
                        prophet_data = time_series.reset_index()
                        prophet_data.columns = ['ds', 'y']
                        
                        # Initialize model
                        prophet_model = Prophet(
                            yearly_seasonality=yearly_seasonality, 
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality
                        )
                        
                        # Add US holidays if selected
                        if include_holidays:
                            prophet_model.add_country_holidays(country_name='US')
                        
                        # Add external regressors
                        if exog_data is not None:
                            for col in exog_data.columns:
                                prophet_data[col] = exog_data[col].values
                                prophet_model.add_regressor(col)
                        
                        # Fit model
                        prophet_model.fit(prophet_data)
                        
                        # Create future dataframe
                        future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='W')
                        
                        # Add external regressors to future data
                        if future_exog is not None:
                            for col in future_exog.columns:
                                # Add to existing dates
                                future.loc[future.index[:len(prophet_data)], col] = prophet_data[col].values
                                
                                # Add to forecast dates
                                for i, date in enumerate(future_exog.index):
                                    idx = len(prophet_data) + i
                                    if idx < len(future):
                                        future.loc[idx, col] = future_exog.loc[date, col]
                        
                        # Make forecast
                        forecast = prophet_model.predict(future)
                        
                        # Extract forecast for the future periods
                        prophet_forecast = forecast.iloc[-forecast_periods:]['yhat']
                        prophet_forecast.index = pd.date_range(
                            start=time_series.index[-1] + pd.Timedelta(days=7), 
                            periods=forecast_periods, 
                            freq='7D'
                        )
                        forecast_results['Prophet'] = prophet_forecast
                        
                        # Save Prophet components for visualization
                        prophet_components = prophet_model.plot_components(forecast)
                        
                    except ImportError:
                        st.warning("Prophet is not installed. Using only ARIMA and SARIMA models.")
                        use_prophet = False
                
                # Create ensemble forecast if selected
                if model_type == "Ensemble":
                    forecasts_df = pd.DataFrame()
                    
                    for name, forecast in forecast_results.items():
                        forecasts_df[name] = forecast
                    
                    if ensemble_method == "Mean":
                        ensemble_forecast = forecasts_df.mean(axis=1)
                    elif ensemble_method == "Median":
                        ensemble_forecast = forecasts_df.median(axis=1)
                    elif ensemble_method == "Weighted":
                        weights = {}
                        if use_arima: weights['ARIMA'] = arima_weight
                        if use_sarima: weights['SARIMA'] = sarima_weight
                        if use_prophet: weights['Prophet'] = prophet_weight
                        
                        # Apply weights
                        weighted_forecasts = pd.DataFrame()
                        for name, weight in weights.items():
                            if name in forecasts_df.columns:
                                weighted_forecasts[name] = forecasts_df[name] * weight
                        
                        ensemble_forecast = weighted_forecasts.sum(axis=1)
                    
                    forecast_results['Ensemble'] = ensemble_forecast
                
                # Determine which forecast to use for display
                if model_type == "ARIMA":
                    final_forecast = forecast_results['ARIMA']
                    model_name = "ARIMA"
                elif model_type == "SARIMA (Seasonal)":
                    final_forecast = forecast_results['SARIMA']
                    model_name = "SARIMA"
                elif model_type == "Prophet":
                    final_forecast = forecast_results['Prophet']
                    model_name = "Prophet"
                else:  # Ensemble
                    final_forecast = forecast_results['Ensemble']
                    model_name = "Ensemble"
                
                # Plot forecast
                st.markdown(f"<div class='section-header'>Sales Forecast ({model_name})</div>", unsafe_allow_html=True)
                
                # Create combined plot
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=time_series.index,
                    y=time_series.values,
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=final_forecast.index,
                    y=final_forecast.values,
                    mode='lines+markers',
                    name='Forecasted Sales',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence intervals for ARIMA/SARIMA
                if model_type in ["ARIMA", "SARIMA (Seasonal)"]:
                    try:
                        # Get prediction intervals
                        if model_type == "ARIMA":
                            pred = model_fit_arima.get_forecast(steps=forecast_periods, exog=future_exog)
                        else:
                            pred = model_fit_sarima.get_forecast(steps=forecast_periods, exog=future_exog)
                            
                        pred_int = pred.conf_int()
                        
                        # Add shaded area for prediction interval
                        fig.add_trace(go.Scatter(
                            x=pred_int.index,
                            y=pred_int.iloc[:, 0],
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            name='Lower 95% CI'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pred_int.index,
                            y=pred_int.iloc[:, 1],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            name='Upper 95% CI'
                        ))
                    except Exception as e:
                        st.warning(f"Could not generate confidence intervals: {e}")
                
                # Add spikes for holidays if available
                if include_features and 'IsHoliday' in feature_cols:
                    holiday_dates = sales_data[sales_data['IsHoliday'] == 1]['Date']
                    
                    # Add vertical lines for holidays
                    for date in holiday_dates:
                        fig.add_vline(
                            x=date, 
                            line_width=1, 
                            line_dash="dash", 
                            line_color="green",
                            annotation_text="Holiday",
                            annotation_position="top right"
                        )
                
                fig.update_layout(
                    title=f'Sales Forecast for Store {selected_store} and Department {selected_dept}',
                    xaxis_title='Date',
                    yaxis_title='Weekly Sales',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast data in table
                st.markdown("<div class='section-header'>Forecast Data</div>", unsafe_allow_html=True)
                forecast_df = final_forecast.reset_index()
                forecast_df.columns = ['Date', 'Forecasted_Sales']
                forecast_df['Forecasted_Sales'] = forecast_df['Forecasted_Sales'].round(2)
                st.dataframe(forecast_df)
                
                # Show model components if using Prophet
                if model_type == "Prophet" and "prophet_components" in locals():
                    st.markdown("<div class='section-header'>Prophet Model Components</div>", unsafe_allow_html=True)
                    st.pyplot(prophet_components)
                
                # Display model summary
                st.markdown("<div class='section-header'>Model Summary</div>", unsafe_allow_html=True)
                
                if model_type == "ARIMA":
                    summary = model_fit_arima.summary()
                    st.text(str(summary))
                elif model_type == "SARIMA (Seasonal)":
                    summary = model_fit_sarima.summary()
                    st.text(str(summary))
                elif model_type == "Ensemble":
                    st.write("Ensemble model using:", ", ".join(forecast_results.keys()))
                    if ensemble_method == "Weighted":
                        st.write("Weights:", {k: round(v, 2) for k, v in weights.items()})
                
            except Exception as e:
                st.error(f"Error creating forecast: {e}")
                st.info("Try adjusting the model parameters or selecting a different store/department combination.")
                st.exception(e)
elif selected_section == "Key Factors Analysis":
    st.markdown("<div class='sub-header'>Key Factors Influencing Sales</div>", unsafe_allow_html=True)
    
    
    selected_store = st.sidebar.selectbox("Select Store", all_stores)
    filtered_depts = sorted(train[train['Store'] == selected_store]['Dept'].unique())
    selected_dept = st.sidebar.selectbox("Select Department", filtered_depts)
    
    
    sales_data = train[(train['Store'] == selected_store) & (train['Dept'] == selected_dept)]
    sales_data = sales_data.sort_values('Date')
    
    
    if len(sales_data) < 10:
        st.warning(f"Not enough data points for Store {selected_store} and Department {selected_dept} for meaningful analysis.")
    else:
       
        st.markdown("<div class='section-header'>Time Series Decomposition</div>", unsafe_allow_html=True)
        
      
        ts_data = sales_data.set_index('Date')['Weekly_Sales']
        
        
        if len(ts_data) >= 14: 
            try:
                
                result = seasonal_decompose(ts_data, model='additive', period=len(ts_data) // 4)
                
                
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
                    vertical_spacing=0.1
                )
                
             
                fig.add_trace(
                    go.Scatter(x=result.observed.index, y=result.observed.values, mode='lines', name='Observed'),
                    row=1, col=1
                )
                
               
                fig.add_trace(
                    go.Scatter(x=result.trend.index, y=result.trend.values, mode='lines', name='Trend'),
                    row=2, col=1
                )
                
              
                fig.add_trace(
                    go.Scatter(x=result.seasonal.index, y=result.seasonal.values, mode='lines', name='Seasonal'),
                    row=3, col=1
                )
                
               
                fig.add_trace(
                    go.Scatter(x=result.resid.index, y=result.resid.values, mode='lines', name='Residual'),
                    row=4, col=1
                )
                
                fig.update_layout(height=800, title_text=f'Time Series Decomposition for Store {selected_store}, Dept {selected_dept}')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error performing time series decomposition: {e}")
                st.info("Try selecting a different store/department combination with more data points.")
        else:
            st.warning("Not enough data points for time series decomposition. Need at least 14 data points.")
        
        # Feature importance analysis
        st.markdown("<div class='section-header'>Feature Importance Analysis</div>", unsafe_allow_html=True)
        
        # Select only numeric columns
        numeric_cols = sales_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target
        exclude_cols = ['Weekly_Sales', 'Store', 'Dept', 'IsHoliday']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation with weekly sales
        correlations = []
        for col in feature_cols:
            if col in sales_data.columns:
                corr = sales_data['Weekly_Sales'].corr(sales_data[col])
                correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create dataframe for visualization
        corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.head(15)  # Top 15 features
        
        # Plot feature importance
        fig = px.bar(
            corr_df,
            x='Feature',
            y='Correlation',
            title=f'Feature Correlation with Weekly Sales (Store {selected_store}, Dept {selected_dept})',
            color='Correlation',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(xaxis_title='Feature', yaxis_title='Correlation', xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 features individual plots
        st.markdown("<div class='section-header'>Top Features vs Weekly Sales</div>", unsafe_allow_html=True)
        
        top_features = corr_df.head(5)['Feature'].tolist()
        
        for feature in top_features:
            if feature in sales_data.columns:
                fig = px.scatter(
                    sales_data,
                    x=feature,
                    y='Weekly_Sales',
                    title=f'Weekly Sales vs {feature}',
                    trendline='ols'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns analysis
        st.markdown("<div class='section-header'>Seasonal Patterns Analysis</div>", unsafe_allow_html=True)
        
        # Monthly analysis
        monthly_sales = sales_data.groupby('Month')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
        monthly_sales.columns = ['Month', 'Average Sales', 'Sales Std Dev']
        
        fig = px.bar(
            monthly_sales,
            x='Month',
            y='Average Sales',
            error_y='Sales Std Dev',
            title=f'Monthly Sales Patterns (Store {selected_store}, Dept {selected_dept})',
            color='Month'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Holiday impact analysis
        st.markdown("<div class='section-header'>Holiday Impact Analysis</div>", unsafe_allow_html=True)
        
        holiday_impact = sales_data.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
        holiday_impact['IsHoliday'] = holiday_impact['IsHoliday'].map({0: 'Non-Holiday', 1: 'Holiday'})
        
        fig = px.bar(
            holiday_impact,
            x='IsHoliday',
            y='Weekly_Sales',
            title=f'Holiday Impact on Sales (Store {selected_store}, Dept {selected_dept})',
            color='IsHoliday'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='insight-box'>
            <h4>📈 Top Influencing Factors for Store {selected_store}, Department {selected_dept}:</h4>
            <ul>
                {''.join([f"<li><strong>{feature}</strong>: Correlation = {corr:.4f}</li>" for feature, corr in correlations[:5]])}
            </ul>
            <p>{'Holiday weeks show ' + ('higher' if holiday_impact.iloc[1]['Weekly_Sales'] > holiday_impact.iloc[0]['Weekly_Sales'] else 'lower') + ' sales compared to non-holiday weeks.'}</p>
            <p>The busiest month appears to be {monthly_sales.iloc[monthly_sales['Average Sales'].argmax()]['Month']}, while the slowest month is {monthly_sales.iloc[monthly_sales['Average Sales'].argmin()]['Month']}.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding-top: 30px;'>
    <p>© 2025 Retail Sales Analysis Dashboard | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)