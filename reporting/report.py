import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import logging

def generate_pdf_report(metrics, report_dir, removed_data_count):
    try:
        pdf_path = os.path.join(report_dir, 'Backtest_Report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # Título
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 50, "Relatório de Backtest")

        # Introdução
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 80)
        text.textLines("""
        Este relatório apresenta os resultados do backtest da estratégia desenvolvida, 
        incorporando as recomendações de Marcos López de Prado para curadoria de dados, 
        engenharia de features, rotulagem, análise de importância das features, 
        validação de modelos e simulação de estratégias de negociação.
        """)
        c.drawText(text)

        # Métricas de Validação Cruzada
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 180, "Métricas de Validação Cruzada")

        c.setFont("Helvetica", 12)
        metrics_text = f"""
        Accuracy Média: {metrics.get('accuracy', 'N/A'):.4f}
        Precision Média: {metrics.get('precision', 'N/A'):.4f}
        Recall Média: {metrics.get('recall', 'N/A'):.4f}
        F1 Score Média: {metrics.get('f1_score', 'N/A'):.4f}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}
        """
        text = c.beginText(50, height - 200)
        text.textLines(metrics_text)
        c.drawText(text)

        # Importância das Features
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 250, "Importância das Features")

        # Gráfico de importância das features para a classe alvo
        feature_img_path = os.path.join(report_dir, f'feature_importance_class_1.png')  # Assuming class 1
        if os.path.exists(feature_img_path):
            c.drawImage(ImageReader(feature_img_path), 50, height - 500, width=500, height=250)
        else:
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 270, f"Gráfico de importância das features para a Classe 1 não disponível.")

        # Backtest Results
        c.setFont("Helvetica-Bold", 14)
        y_backtest = height - 520
        c.drawString(50, y_backtest, "Resultados do Backtest")

        # Gráfico de Retornos Cumulativos
        cumulative_returns_path = os.path.join(report_dir, 'cumulative_returns.png')
        if os.path.exists(cumulative_returns_path):
            c.drawImage(ImageReader(cumulative_returns_path), 50, y_backtest - 250, width=500, height=250)
        else:
            c.setFont("Helvetica", 12)
            c.drawString(50, y_backtest - 40, "Gráfico de retornos cumulativos não disponível.")

        # Informações sobre registros removidos
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_backtest - 300, "Registros Removidos Durante a Curadoria de Dados")

        c.setFont("Helvetica", 12)
        removed_text = f"Total de registros removidos: {removed_data_count}"
        text = c.beginText(50, y_backtest - 320)
        text.textLines(removed_text)
        c.drawText(text)

        c.save()
        logging.info(f"Relatório PDF gerado em: {pdf_path}")
        print(f"Relatório PDF gerado em: {pdf_path}")
    except Exception as e:
        logging.error(f"Erro durante a geração do relatório em PDF: {e}")
        raise
