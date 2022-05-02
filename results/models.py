from django.db import models

class Results(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    full_name = models.CharField(max_length=500, blank=True, default='')
    email = models.CharField(max_length=500)
    is_healthy = models.BooleanField(default=False)

    class Meta:
        ordering = ['created']

