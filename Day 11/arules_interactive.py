import streamlit as st 
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os
os.chdir(r"C:\Users\dbda\Desktop\MACHINE LEARNING\Datasets")


st.title("Association Rules")

from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
groceries=[]
with open("Groceries.csv","r") as f:groceries=f.read()
groceries =groceries.split("\n")

groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))


te=TransactionEncoder()
te_ary=te.fit(groceries_list).transform(groceries_list)


fp_df=pd.DataFrame(te_ary,columns=te.columns_)





# fp_df = pd.read_csv("Cosmetics.csv",index_col=0)
# fp_df = fp_df.astype(bool)

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