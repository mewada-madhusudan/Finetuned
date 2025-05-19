
import gradio as gr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import pandas as pd

font_path="simfang.ttf"
lang_list= ['ch', 'en', "japan"]



header='''
<header style="width: 100%; text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #6a11cb, #2575fc); color: white; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
  <h1 style="font-size: 2.5rem; margin-bottom: 15px;">Paddle OCR</h1>
  <p style="font-size: 1.2rem; margin-bottom: 25px; line-height: 1.5; max-width: 800px; margin-left: auto; margin-right: auto;">
    An unofficial demo of Paddle OCR. Part of my 
    <a href="https://huggingface.co/spaces/holyhigh666/graph-digitizer" target="_blank" style="color: white; text-decoration: underline;">
      Graph Digitizer
    </a> project.
  </p>
</header>

'''

def perform_ocr(img, lang):
    """
    Perform OCR on the given image and display results.
    """
    #ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    # Initialize the PaddleOCR model
    if lang == "en":
        ocr = PaddleOCR(use_angle_cls=True, lang=lang,
                    det_model_dir=f"models/{lang}_PP-OCRv3_det_infer",
                    rec_model_dir=f"models/{lang}_PP-OCRv4_rec_infer",
                    cls_model_dir="models/ch_ppocr_mobile_v2.0_cls_infer",
                   )
    elif lang == "ch":        
        ocr = PaddleOCR(use_angle_cls=True, lang=lang,
                        det_model_dir=f"models/{lang}_PP-OCRv4_det_infer",
                        rec_model_dir=f"models/{lang}_PP-OCRv4_rec_infer",
                        cls_model_dir="models/ch_ppocr_mobile_v2.0_cls_infer",
                       )
    elif lang == "japan":
        ocr = PaddleOCR(use_angle_cls=True, lang=lang,
                        det_model_dir="models/ch_PP-OCRv4_det_infer",
                        rec_model_dir="models/japan_mobile_v2.0_rec_infer",
                        cls_model_dir="models/ch_ppocr_mobile_v2.0_cls_infer",
                       )
    else:
        return "Error, selected language does not exist "
    results = ocr.ocr(img, cls=True)

    if not results:
        print("No text detected in the image.")
        return None, None

    result = results[0]
    boxes = [line[0] for line in result]
    texts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    image = Image.fromarray(img).convert('RGB')
    annotated_image = draw_ocr(image, boxes, font_path="simfang.ttf")
    annotated_image = Image.fromarray(annotated_image)

    df = pd.DataFrame({ 'texts': texts, 'boxes': boxes,'scores': scores})
    combined_string = "".join(texts)
    return annotated_image, df, combined_string



def create_app():
    with gr.Blocks() as app:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    image_input = gr.Image(#type="array",
                                           height=500,
                                           width=500,
                                           label= "Upload Image",
                                                 )
                with gr.Row():
                    select_lang=gr.Dropdown(choices=lang_list, label="Select Language",value= "ch")
                submit_button = gr.Button("Process")
            with gr.Column(scale=5):
                with gr.Row():
                    output_image = gr.Image(height=500,
                                            width=500,
                                            label="Text with Bbox")
                with gr.Row():
                    output_csv = gr.DataFrame(headers=['texts', 'boxes','scores'],
                                              show_copy_button=True,
                                              show_row_numbers=True,
                                              label="OCR Result"
                                              )
                with gr.Row():
                    combined_text=gr.Textbox(label="Combined Texts",show_copy_button=True)
        submit_button.click(
            perform_ocr,
            inputs=[image_input, select_lang],
            outputs=[output_image, output_csv,combined_text]
        )
        examples = gr.Examples(
        examples=[
            ["image/example2.png", "en"],
        ],
        inputs=[image_input, select_lang],
    )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()