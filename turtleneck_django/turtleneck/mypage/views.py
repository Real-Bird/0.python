
#from blog.views import PostList
from single_pages.models import User
# from django.http import request, response
from django.shortcuts import render, redirect
# from django.views.generic.detail import DetailView
# from django.views.generic import CreateView ,ListView, UpdateView
# from django.contrib.auth.mixins import LoginRequiredMixin
# from django.core.exceptions import PermissionDenied
# from django.utils.text import slugify
# from django.shortcuts import get_object_or_404
# from .models import Category, Tag , MyPost
from blog.models import Post

# from django.db.models import Q


def mypage(request):
    User.objects = User.objects.filter(user_name = request.session['loginuser'] ) 
    Post.objects = Post.objects.filter(author = User.objects[0].id )
    my_post = Post.objects.order_by('-pk')[:5]
    
    return render(
        request, 
        'mypage/mypage.html',
        {
            'my_post': my_post,
        }
    )



# class MyPostUpdate(LoginRequiredMixin, UpdateView):
#     model= MyPost
#     fields = ['title', 'hook_text', 'content', 'head_image', 'file_upload','category']

#     template_name = 'mypage/mypost_update_form.html'

#     def get_context_data(self, **kwargs):
#         context = super(MyPostUpdate, self).get_context_data()
#         if self.object.tags.exists():
#             tags_str_list = list()
#             for t in self.object.tags.all():
#                 tags_str_list.append(t.name)
#             context['tags_str_default']='; '.join(tags_str_list)
#         return context
    

#     def dispatch(self, request, *args, **kwargs):
#         if request.user.is_authenticated and request.user == self.get_object().author:
#             return super(MyPostUpdate, self).dispatch(request, *args, **kwargs)
#         else:
#             raise PermissionDenied

#     def form_valid(self, form):
#         response = super(MyPostUpdate, self).form_valid(form)
#         self.object.tags.clear()

#         tags_str = self.request.POST.get('tags_str')
#         if tags_str:
#             tags_str = tags_str.strip()
#             tags_str = tags_str.replace(',',';')
#             tags_list = tags_str.split(';')

#             for t in tags_list:
#                 t = t.strip()
#                 tag, is_tag_created = Tag.objects.get_or_create(name=t)
#                 if is_tag_created:
#                     tag.slug = slugify(t, allow_unicode=True)
#                     tag.save()
#                 self.object.tags.add(tag)
#         return response

# class MyPostCreate(LoginRequiredMixin,  CreateView):
#     model = MyPost
#     fields = ['title', 'hook_text', 'content', 'head_image', 'file_upload', 'category']

#     def test_func(self):
#         return self.request.user.is_superuser or self.request.user.is_staff

#     def form_valid(self, form):
#         current_user = self.request.user
#         if current_user.is_authenticated :
#             form.instance.author = current_user
#             response = super(MyPostCreate, self).form_valid(form)

#             tags_str = self.request.POST.get('tags_str')
#             if tags_str:
#                 tags_str = tags_str.strip()

#                 tags_str = tags_str.replace(',', ';')
#                 tags_list = tags_str.split(';')

#                 for t in tags_list:
#                     t = t.strip()
#                     tag, is_tag_created = Tag.objects.get_or_create(name=t)
#                     if is_tag_created:
#                         tag.slug = slugify(t, allow_unicode=True)
#                         tag.save()
#                     self.object.tags.add(tag)

#             return response

#         else:
#                 return redirect('/mypage/')
       


# class MyPostList(ListView):
#     model = MyPost
#     ordering = '-pk'
#     paginate_by = 5
    
#     def get_context_data(self, **kwargs):
#         context = super(MyPostList,self).get_context_data()
#         context["mypost_list"] = MyPost.objects.filter( author =  self.request.user)
    
#         context['categories'] = Category.objects.all()
#         context['no_category_post_count'] = MyPost.objects.filter(category=None).count()
#         return context





# class MyPostSearch(MyPostList):
#     paginate_by = None

#     def get_queryset(self):
#         q = self.kwargs['q']
#         mypost_list = MyPost.objects.filter(
#             Q(title__contains=q)|Q(tags__name__contains=q)
#         ).distinct()
#         return mypost_list
    
#     def get_context_data(self, **kwargs):
#         context = super(MyPostSearch, self).get_context_data()
#         q = self.kwargs['q']
#         context['search_info'] = f'Search: {q} ({self.get_queryset().count()})'
#         return context



# class MyPostDetail(DetailView):
#     model = MyPost
#     template_name = 'mypage/mypage_detail.html'

#     def get_context_data(self, **kwargs):
#         context = super(MyPostDetail, self).get_context_data()
#         context['categories'] = Category.objects.all()
#         context['no_category_post_count'] = MyPost.objects.filter(category=None).count()
 
#         return context

# def category_page(request, slug):
#     if slug == ' no_category':
#         category = '기타'
#         mypost_list = MyPost.objects.filter(category=None)
#     else:
#         category = Category.objects.get(slug=slug)
#         mypost_list = MyPost.objects.filter(category=category)
        
#     return render(
#         request, 
#         'mypage/mypost_list.html',
#         {
#             'mypost_list':mypost_list,
#             'categories':Category.objects.all(),
#             'no_category_post_count': MyPost.objects.filter(category=None).count(),
#             'category': category,
#         }
#     )

# def tag_page(request, slug):
#     tag = Tag.objects.get(slug = slug)
#     mypost_list = tag.mypost_set.all()

#     return render(
#         request, 
#         'mypage/mypost_list.html',
#         {
#             'mypost_list': mypost_list,
#             'tag':tag,
#             'categories': Category.objects.all(),
#             'no_category_post_count': MyPost.objects.filter(category=None).count(),
#         }
#         )  



# def delete_mypost(request, pk):
#     mypost = get_object_or_404(MyPost, pk=pk)

#     if request.user.is_authenticated and request.user == mypost.author:
#         mypost.delete()
#         return render(
#         request, 
#         'mypage/mypost_list.html',
#         )  
#     else:
#         raise PermissionDenied

