from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from results.models import Results
from results.serializers import ResultsSerializer
from algo.trained import DatasetCreator


@csrf_exempt
def snippet_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = Results.objects.all()
        serializer = ResultsSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        files = request.FILES
        temp_dict = request.POST.copy()

        if not files:
            return JsonResponse({'error': 'No file to check'}, status=400)

        trained_model = DatasetCreator()
        prediction_result = trained_model.predict(files['sound'])
        temp_dict['is_healthy'] = prediction_result['success']
        serializer = ResultsSerializer(data=temp_dict)

        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)


@csrf_exempt
def snippet_detail(request, pk):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        snippet = Results.objects.get(pk=pk)
    except Results.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = ResultsSerializer(snippet)
        return JsonResponse(serializer.data)

    elif request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = ResultsSerializer(snippet, data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        snippet.delete()
        return HttpResponse(status=204)
