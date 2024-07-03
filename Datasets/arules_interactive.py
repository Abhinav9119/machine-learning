import streamlit as st 
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.title("Association Rules")

fp_df = pd.read_csv("Cosmetics.csv",index_col=0)
fp_df = fp_df.astype(bool)

try: 
    support = st.slider("Support:",value=0.01, min_value=0.001,
                              step=0.001,max_value=1.)
    conf = st.slider("Confidence:",value=0.5, min_value=0.1,step=0.01,
                              max_value=1.)
    lr = st.number_input("Lift Ratio:",value=1.1)


    itemsets = apriori(fp_df, min_support=support,
                       use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', 
                              min_threshold=conf)
    
    
    st.dataframe(rules)
except ValueError:
    st.text("No Rules Generated")