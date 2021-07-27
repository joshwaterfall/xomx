import requests
import pandas as pd
from io import StringIO
from IPython import embed as e

assert e


def gdc_create_manifest(disease_type, project_list, nr_of_cases_list):
    df_list = []
    for proj, nr_of_cases in zip(project_list, nr_of_cases_list):
        fields = [
            "file_name",
            "md5sum",
            "file_size",
            "state",
            "cases.project.project_id",
        ]
        fields = ",".join(fields)
        files_endpt = "https://api.gdc.cancer.gov/files"
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {"field": "cases.project.project_id", "value": [proj]},
                },
                {
                    "op": "in",
                    "content": {"field": "cases.disease_type", "value": [disease_type]},
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": ["Transcriptome Profiling"],
                    },
                },
                {
                    "op": "in",
                    "content": {"field": "files.type", "value": ["gene_expression"]},
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.analysis.workflow_type",
                        "value": ["HTSeq - Counts"],
                    },
                },
                {
                    "op": "in",
                    "content": {"field": "files.data_format", "value": ["TXT"]},
                },
            ],
        }
        params = {
            "filters": filters,
            "fields": fields,
            "format": "TSV",
            "size": str(nr_of_cases),
        }
        response = requests.post(
            files_endpt, headers={"Content-Type": "application/json"}, json=params
        )
        df = pd.read_table(StringIO(response.content.decode("utf-8")))
        df = df.rename(
            columns={
                "file_name": "filename",
                "file_size": "size",
                "md5sum": "md5",
                "cases.0.project.project_id": "annotation",
            }
        )
        df = df[["id", "filename", "md5", "size", "state", "annotation"]]
        df_list.append(df)
    return df_list
