
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('', include('calc.urls')),
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    #path('', include('final.urls'))
]

urlpatterns = urlpatterns + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)