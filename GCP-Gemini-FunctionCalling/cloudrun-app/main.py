import vertexai
from flask import Flask, request, jsonify
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    GenerationConfig,
    Part,
    Tool,
    GenerationResponse
)
from typing import Any, Dict, List
from google.cloud import bigquery
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROJECT_ID = "no-latency-labs"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

client = bigquery.Client()

query_product_info = FunctionDeclaration(
    name="query_product_info",
    description="Fetch the product info from a bigquery table",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Name of the product for which the details are to be fetched",
            }
        },
    },
)

get_weekly_product_orders = FunctionDeclaration(
    name="get_weekly_product_orders",
    description="Fetch the total orders for the last 1 week for any given product",
    parameters={
        "type": "object",
        "properties": {
           "product_name": {
                "type": "string",
                "description": "Name of the product for which the details are to be fetched",
            }
        },
    },
)

fetch_info_tool = Tool(
    function_declarations=[
        query_product_info,
        get_weekly_product_orders
    ],
)

gemini_model = GenerativeModel(
    "gemini-1.5-pro-001",
    generation_config=GenerationConfig(temperature=0),
    tools=[fetch_info_tool],
)

def extract_function_calls(response: GenerationResponse) -> List[Dict]:
    function_calls: List[Dict] = []
    if response.candidates[0].function_calls:
        for function_call in response.candidates[0].function_calls:
            function_call_dict = {'function_name': function_call.name}
            for key, value in function_call.args.items():
                if isinstance(value, dict):
                    first_value_key = next(iter(value))
                    arg_value = value[first_value_key]
                else:
                    arg_value = value
                function_call_dict['arg_value'] = arg_value
                
            function_calls.append(function_call_dict)
            
    return function_calls

def query_product_info(product_name):
    
    product_name = product_name.lower().strip()
    
    query = f"""
    SELECT distinct name, brand, ROUND(retail_price, 1) AS price, department
    FROM `bigquery-public-data.thelook_ecommerce.products`
    WHERE LOWER(name) LIKE @product_name
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product_name", "STRING", f"%{product_name.lower()}%")
        ]
    )

    # Run the query
    query_job = client.query(query, job_config=job_config)

    # Wait for the query to finish
    results = query_job.result()

    # Process the results
    response = []
    for row in results:
        response.append({
            "name": row.name,
            "brand": row.brand,
            "price": row.price,
            "department": row.department
        })

    return response

def get_weekly_product_orders(product_name):

    product_name = product_name.lower().strip()
    query = f"""
        select a.name as product_name,b.total_orders from 
        (SELECT id,name,brand,round(retail_price,1) as price,department FROM `bigquery-public-data.thelook_ecommerce.products` 
        where lower(name) like '%{product_name}%') a
        join 
        (
          SELECT product_id,sum(distinct order_id) as total_orders FROM `bigquery-public-data.thelook_ecommerce.order_items` 
        where date(created_at)>=current_date-7 group by 1
        ) b
        on a.id = b.product_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("product_name", "STRING", f"%{product_name.lower()}%")
        ]
    )

    # Run the query
    query_job = client.query(query, job_config=job_config)

    # Wait for the query to finish
    results = query_job.result()
    
    results = list(query_job.result()) 
    
    if results is None or not results:
        response = [{'product_name':product_name, 'total_orders': 0}]
    else:
        response = []
        for row in results:
            response.append({
                "product_name": row.product_name,
                "total_orders": row.total_orders
            })

    return response

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    
    data = request.json
    user_input = data.get('prompt', '')
    chat = gemini_model.start_chat()

    prompt = user_input + "\nGive a concise, high-level summary. Only use information that you learn from the API response or BigQuery depending on the question. Do not make up any information."
    response = chat.send_message(prompt)

    function_calls = extract_function_calls(response)

    api_response = []
    for function_call in function_calls:
        if function_call['function_name'] == 'query_product_info':
            result = query_product_info(function_call['arg_value'])
            result.append(function_call['function_name'])
            api_response.append(result)
        elif function_call['function_name'] == 'get_weekly_product_orders':
            result = get_weekly_product_orders(function_call['arg_value'])
            result.append(function_call['function_name'])
            api_response.append(result)

    response_parts = []
    for item in api_response:
        try:
            part = Part.from_function_response(
                name=item[1],  # Function name
                response={"content": item[0]}  # Response content
            )
            response_parts.append(part)
        except IndexError as e:
            logger.error(f"Error processing item {item}: {e}")

    final_response = chat.send_message(response_parts)

    return jsonify({'response': str(final_response.text)})

if __name__ == '__main__':
    app.run(debug=True)