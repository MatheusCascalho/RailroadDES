import xml.etree.ElementTree as ET
import plotly.express as px

# Parse do XML de cobertura
tree = ET.parse('coverage.xml')
root = tree.getroot()

files = []
coverage = []

for packages in root.findall('packages'):
    for package in packages.findall('package'):
        classes = package.find('classes')
        if classes is not None:
            for clazz in classes.findall('class'):
                name = clazz.attrib.get('filename')
                line_rate = float(clazz.attrib.get('line-rate', 0))
                files.append(name)
                coverage.append(line_rate)

# Criar dataframe para Plotly
import pandas as pd

df = pd.DataFrame({
    'file': files,
    'coverage': coverage,
    'coverage_percent': [c * 100 for c in coverage]
})

# Gerar treemap com Plotly
fig = px.treemap(
    df,
    path=['file'],                 # Hierarquia (aqui só arquivo)
    values='coverage_percent',     # Tamanho dos blocos
    color='coverage_percent',      # Cor pelo percentual
    color_continuous_scale='RdYlGn',  # Verde para alta cobertura, vermelho para baixa
    title='Treemap da Cobertura de Código'
)

fig.show()

fig.write_html("coverage_treemap.html")

