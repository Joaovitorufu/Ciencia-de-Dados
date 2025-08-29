import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Bases de dados/KaggleV2-May-2016.csv', header=0)

#verificando se existe atributos invalidos.
print(df['Age'].max())
print(df['Age'].min())
#removendo as idades invalidas
df = df[df['Age'] >= 0]
#removendo atributos de controle de index
df=df.drop(columns=['PatientId','AppointmentID'])
#mapeando a coluna gender para atributos numéricos
df['Gender'] = df['Gender'].map({'F':0,'M':1})
#retirando o Timezone
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.tz_localize(None)
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.tz_localize(None)
#juntando as 2 colunas scheduledDay e AppointementDay em uma coluna unica WaitingTime(tempo de espera) medido e dias
df['WaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
print(df['WaitingTime'].max())
print(df['WaitingTime'].min())
print(df.head(5))

