from tqdm import tqdm
import requests

files = [("green_tripdata_2022-01.parquet", "."), ("green_tripdata_2021-01.parquet", "./evidently_service/datasets")]
# files = [("fhv_tripdata_2022-01.parquet", "."), ("fhv_tripdata_2021-01.parquet", "./evidently_service/datasets")]
print(f"Download files:")
for file, path in files:
    # url = f"https://nyc-tlc.s3.amazonaws.com/trip+data/{file}"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    # url = f"https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/{file}"
    resp = requests.get(url, stream=True)
    save_path = f"{path}/{file}"
    with open(save_path, "wb") as handle:
        for data in tqdm(resp.iter_content(),
                         desc=f"{file}",
                         postfix=f"save to {save_path}",
                         total=int(resp.headers["Content-Length"])):
            handle.write(data)



# url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_2021-01.parquet'
# df = pd.read_parquet(url)
