# 🤖 Consultor de Inversiones IA - Sistema Experto RAG & Quant Finance

Este proyecto es un **Agente de Inteligencia Artificial de nivel Senior** diseñado para la auditoría y el análisis de riesgo financiero. Combina el procesamiento de lenguaje natural (RAG) con modelos matemáticos determinísticos para ofrecer análisis financieros precisos, auditables y conectados a datos reales de mercado.

## 🚀 Funcionalidades Clave

### 1. Motor RAG (Retrieval-Augmented Generation)
* **Ingesta Inteligente:** Procesa documentos de más de 200 páginas (Form 10-K, Reportes Anuales) mediante segmentación semántica profesional.
* **Cero Alucinaciones:** El sistema responde basándose exclusivamente en el contexto del documento y cita la **página exacta** de la fuente para validación humana.
* **Memoria Vectorial:** Implementación de **FAISS** para búsquedas semánticas de alta dimensionalidad y recuperación ultrarrápida.

### 2. Análisis Quant & Riesgo (Finance Engine)
* **Simulación Monte Carlo:** Genera miles de escenarios futuros basados en la volatilidad histórica de activos para proyectar rendimientos y riesgos.
* **Modelo de Default Merton:** Calcula la probabilidad real de impago (Default) tratando la estructura de capital de la empresa como una opción financiera.
* **Extracción ETL Automática:** Identificación y estructuración de métricas clave como Deuda Total, Caja y Pasivos directamente desde el texto no estructurado.

### 3. Datos en Tiempo Real
* **Integración con Yahoo Finance:** Conecta el análisis de documentos estáticos con precios de mercado y datos históricos en vivo para una visión 360°.



## 🛠️ Stack Tecnológico
* **IA/NLP:** LangChain, OpenAI API (GPT-4o / GPT-3.5-turbo).
* **Vector Database:** FAISS (Facebook AI Similarity Search).
* **Finanzas Cuantitativas:** Pandas, NumPy, Scipy, yFinance.
* **Infraestructura:** Arquitectura modular desacoplada en Python.

## 📁 Estructura del Proyecto
- `rag/`: Lógica de recuperación, manejo de prompts y cadena de QA.
- `finance/`: Motores de cálculo (Monte Carlo, Merton) y extracción de deuda.
- `ingest/`: Procesamiento de PDFs, limpieza de texto y carga de metadatos.
- `vectorstore/`: Construcción y gestión del índice vectorial.
- `app.py`: Orquestador principal y lógica del sistema experto.

## ⚙️ Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/Consultor-Inversiones-IA.git](https://github.com/tu-usuario/Consultor-Inversiones-IA.git)
Instalar dependencias:

Bash
pip install -r requirements.txt
Configurar variables de entorno: Crea un archivo .env en la raíz del proyecto (usa .env.example como guía) e inserta tu API Key:

Fragmento de código
OPENAI_API_KEY=tu_clave_aqui
4 Ejecutar el sistema:

Bash
qa_engine.py

## Demo (Video)
🎥 https://youtu.be/IJZgELb1eyM
