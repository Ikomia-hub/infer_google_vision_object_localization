import copy
from ikomia import core, dataprocess
from google.cloud import vision
import os
import io
import cv2


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGoogleVisionObjectLocalizationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.conf_thres = 0.3
        self.google_application_credentials = ''

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        self.conf_thres = float(params["conf_thres"])
        self.google_application_credentials = str(params["google_application_credentials"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["conf_thres"] = str(self.conf_thres)
        params["google_application_credentials"] = str(self.google_application_credentials)
        return params

# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGoogleVisionObjectLocalization(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferGoogleVisionObjectLocalizationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()

        # Get input
        input = self.get_input(0)
        src_image = input.get_image()

        # Set output
        output_dict = self.get_output(2)

        # Get parameters
        param = self.get_param_object()

        if self.client is None:
            if param.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = param.google_application_credentials
            self.client = vision.ImageAnnotatorClient()

        # Convert the NumPy array to a byte stream
        height_ini, width_ini = src_image.shape[:2]
        src_image = src_image[..., ::-1] # Convert to bgr
        is_success, image_buffer = cv2.imencode(".jpg", src_image)
        byte_stream = io.BytesIO(image_buffer)

        response = self.client.object_localization(image=byte_stream)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        objects = response.localized_object_annotations

        if len(objects)==0:
            print('No object detected')

        # Process output
        for i, object_ in enumerate(objects):
            if object_.score < param.conf_thres: # skip detections with lower score
                continue
            # Get box coordinates
            vertices = [(vertex.x,vertex.y) for vertex in object_.bounding_poly.normalized_vertices]
            x_box = vertices[0][0] * width_ini
            y_box = vertices[0][1] * height_ini
            w = (vertices[1][0] - vertices[0][0]) * width_ini
            h = (vertices[2][1] - vertices[0][1]) * height_ini

            self.set_names([f'{object_.name}'])

            # Display graphics
            self.add_object(i, 0, object_.score, float(x_box), float(y_box), w, h)

        output_dict.data = {'logo_annotations': str(response.localized_object_annotations)}

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGoogleVisionObjectLocalizationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_google_vision_object_localization"
        self.info.short_description = "The Vision API can detect and extract multiple objects in an image with Object Localization."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Google"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://cloud.google.com/vision/docs/object-localizer"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_google_vision_object_localization"
        self.info.original_repository = "https://github.com/googleapis/python-vision"
        # Keywords used for search
        self.info.keywords = "Logo detection,Google,Cloud,Vision AI"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferGoogleVisionObjectLocalization(self.info.name, param)
