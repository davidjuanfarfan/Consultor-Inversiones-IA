# Consultor-Inversiones-IA
Agente de IA (RAG) para Auditoría Financiera y Documental. Procesa PDFs extensos con precisión del 100%, realiza análisis comparativos multianuales y cita la página exacta de la fuente para eliminar alucinaciones. Arquitectura modular en Python con memoria vectorial.
# 🤖 Consultor de Inversiones IA - Motor RAG Profesional

Este sistema es un **Agente de Inteligencia Artificial** diseñado para auditar y analizar documentos financieros complejos (como reportes 10-K de Tesla) con precisión quirúrgica.

## 🚀 Capacidades Destacadas
* **Cero Alucinaciones:** El sistema cita la **página exacta** de la fuente para cada respuesta.
* **Análisis Financiero Avanzado:** Realiza cálculos de deuda, ingresos y simulaciones de riesgo automáticamente.
* **Procesamiento Universal:** Capaz de leer PDFs de 200+ páginas gracias a su base de datos vectorial (FAISS/Chroma).
* **Arquitectura Modular:** Separación clara entre ingesta de datos, lógica financiera y motor de IA.

## 🛠️ Tecnologías
* **IA:** OpenAI API / LangChain (RAG)
* **Datos:** FAISS (Vector Store) para búsqueda semántica.
* **Finanzas:** Simulaciones de Monte Carlo y análisis de ratios en Python.
* **Interfaz:** Terminal interactiva (próximamente Web Dashboard).

## 📁 Estructura del Proyecto
- `rag/`: Cerebro de la IA y manejo de contexto.
- `finance/`: Motores de cálculo y simulaciones.
- `ingest/`: Procesamiento y limpieza de PDFs.
- `vectorstore/`: Almacenamiento de memoria a largo plazo.
