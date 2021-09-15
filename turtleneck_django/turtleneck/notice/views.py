from django.http import request, response
from django.shortcuts import render, redirect
from django.views.generic.detail import DetailView
from django.views.generic import CreateView ,ListView, UpdateView
# from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied

from .models import Category, Notice 






class NoticeUpdate(UpdateView):
    model= Notice
    fields = ['title', 'hook_text', 'content', 'head_image', 'file_upload','category']

    template_name = 'notice/notice_update_form.html'

    def get_context_data(self, **kwargs):
        context = super(NoticeUpdate, self).get_context_data()
        return context
    

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated and request.user == self.get_object().author:
            return super(NoticeUpdate, self).dispatch(request, *args, **kwargs)
        else:
            raise PermissionDenied

    def form_valid(self, form):
        response = super(NoticeUpdate, self).form_valid(form)
        self.object.tags.clear()
        return response

    

class NoticeCreate(CreateView):
    model = Notice
    fields = ['title', 'hook_text', 'content', 'head_image', 'file_upload', 'category']

    def test_func(self):
        return self.request.user.is_superuser or self.request.user.is_staff

    def form_valid(self, form):
        current_user = self.request.user
        if current_user.is_authenticated and (current_user.is_staff or current_user.is_superuser):
            form.instance.author = current_user
            response = super(NoticeCreate, self).form_valid(form)
            return response

        else:
                return redirect('/notice/')


class NoticeList(ListView):
    model = Notice
    ordering = '-pk'
    paginate_by = 5

    def get_context_data(self, **kwargs):
        context = super(NoticeList,self).get_context_data()
        context['categories'] = Category.objects.all()
        context['no_category_notice_count'] = Notice.objects.filter(category=None).count()
        return context





class NoticeDetail(DetailView):
    model = Notice
    template_name = 'notice/notice_detail.html'

    def get_context_data(self, **kwargs):
        context = super(NoticeDetail, self).get_context_data()
        context['categories'] = Category.objects.all()
        context['no_category_notice_count'] = Notice.objects.filter(category=None).count()
        return context



def category_page(request, slug):
    if slug == 'no_category':
        category = '기타'
        notice_list = Notice.objects.filter(category=None)
        print(notice_list)
    else:
        category = Category.objects.get(slug=slug)
        
        notice_list = Notice.objects.filter(category=category)
        
    
    return render(
        request,
        'notice/notice_list.html',
        {
            'notice_list': notice_list,
            'categories': Category.objects.all(),
            'no_category_notice_count': Notice.objects.filter(category=None).count(),
            'category': category,
        }
    )








# Create your views here.
