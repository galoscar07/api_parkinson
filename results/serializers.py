from rest_framework import serializers
from results.models import Results


class ResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Results
        fields = ['id', 'created', 'full_name', 'email', 'is_healthy']
