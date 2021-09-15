
from django.http import request, response
from django.shortcuts import render, redirect
from django.views.generic.detail import DetailView
from django.views.generic import CreateView ,ListView, UpdateView
# from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied
from django.utils.text import slugify
from django.shortcuts import get_object_or_404
from .models import Category, Post , Tag , Comment
from .forms import CommentForm
from django.db.models import Q
from single_pages.models import User




class PostUpdate(UpdateView):
    model= Post
    fields = ['title', 'hook_text', 'content', 'head_image', 'file_upload','category']

    template_name = 'blog/post_update_form.html'

    def get_context_data(self, **kwargs):
        context = super(PostUpdate, self).get_context_data()
        if self.object.tags.exists():
            tags_str_list = list()
            for t in self.object.tags.all():
                tags_str_list.append(t.name)
            context['tags_str_default']='; '.join(tags_str_list)
        return context
    

    def dispatch(self, request, *args, **kwargs):
        # if request.session.loginuser == self.get_object().author:
            return super(PostUpdate, self).dispatch(request, *args, **kwargs)
        # else:
        #     raise PermissionDenied

    def form_valid(self, form):
        response = super(PostUpdate, self).form_valid(form)
        self.object.tags.clear()

        tags_str = self.request.POST.get('tags_str')
        if tags_str:
            tags_str = tags_str.strip()
            tags_str = tags_str.replace(',',';')
            tags_list = tags_str.split(';')

            for t in tags_list:
                t = t.strip()
                tag, is_tag_created = Tag.objects.get_or_create(name=t)
                if is_tag_created:
                    tag.slug = slugify(t, allow_unicode=True)
                    tag.save()
                self.object.tags.add(tag)
        return response

class PostCreate(CreateView):
    model = Post
    fields = ['title', 'hook_text', 'content', 'head_image', 'file_upload', 'category']

    def form_valid(self, form):
        us = User.objects.filter(user_name = self.request.session['loginuser'] )

        if self.request.session['loginuser'] :
            form.instance.author = us[0]
            response = super(PostCreate, self).form_valid(form)

        tags_str = self.request.POST.get('tags_str')
        if tags_str:
            tags_str = tags_str.strip()

            tags_str = tags_str.replace(',', ';')
            tags_list = tags_str.split(';')

            for t in tags_list:
                t = t.strip()
                tag, is_tag_created = Tag.objects.get_or_create(name=t)
                if is_tag_created:
                    tag.slug = slugify(t, allow_unicode=True)
                    tag.save()
                self.object.tags.add(tag)

        return response

        # else:
        #     return redirect('/blog/')


class PostList(ListView):
    model = Post
    ordering = '-pk'
    paginate_by = 5

    def get_context_data(self, **kwargs):
        context = super(PostList,self).get_context_data()
        context['categories'] = Category.objects.all()
        context['no_category_post_count'] = Post.objects.filter(category=None).count()
        return context

class MyPostList(ListView):
    model = Post
    ordering = '-pk'
    paginate_by = 5

    def get_context_data(self, **kwargs):
        context = super(MyPostList,self).get_context_data()
        # User.objects = User.objects.filter(user_name = request.session['loginuser'] ) 
        context['post_list'] = Post.objects.filter(author=User.objects[0].id)
        context['categories'] = Category.objects.all()
        context['no_category_post_count'] = Post.objects.filter(category=None).count()
        return context


class PostSearch(PostList):
    paginate_by = None

    def get_queryset(self):
        q = self.kwargs['q']
        post_list = Post.objects.filter(
            Q(title__contains=q)|Q(tags__name__contains=q)
        ).distinct()
        return post_list
    
    def get_context_data(self, **kwargs):
        context = super(PostSearch, self).get_context_data()
        q = self.kwargs['q']
        context['search_info'] = f'Search: {q} ({self.get_queryset().count()})'
        return context



class PostDetail(DetailView):
    model = Post
    template_name = 'blog/single_post_page.html'

    def get_context_data(self, **kwargs):
        context = super(PostDetail, self).get_context_data()
        context['categories'] = Category.objects.all()
        context['no_category_post_count'] = Post.objects.filter(category=None).count()
        context['comment_form'] = CommentForm
        return context



def category_page(request, slug):
    if slug == 'no_category':
        category = '기타'
        post_list = Post.objects.filter(category=None)
    else:
        category = Category.objects.get(slug=slug)
        post_list = Post.objects.filter(category=category)

    return render(
        request,
        'blog/post_list.html',
        {
            'post_list': post_list,
            'categories': Category.objects.all(),
            'no_category_post_count': Post.objects.filter(category=None).count(),
            'category': category,
        }
    )


def tag_page(request, slug):
    tag = Tag.objects.get(slug = slug)
    post_list = tag.post_set.all()

    return render(
        request, 
        'blog/post_list.html',
        {
            'post_list': post_list,
            'tag':tag,
            'categories': Category.objects.all(),
            'no_category_post_count': Post.objects.filter(category=None).count(),
        }
        )  



def new_comment(request, pk):
    us = User.objects.filter(user_name = request.session['loginuser'] )

    if request.session['loginuser'] :
        post = get_object_or_404(Post, pk=pk)

        if request.method == 'POST':
            comment_form = CommentForm(request.POST)
            if comment_form.is_valid():
                comment = comment_form.save(commit=False)
                comment.post = post
                comment.author = us[0]
                comment.save()
                return redirect(comment.get_absolute_url())
            else:
                return redirect(post.get_absolute_url())
    else:
        raise PermissionDenied


class CommentUpdate(UpdateView):
    model = Comment
    form_class = CommentForm

    def dispatch(self, request, *args, **kwargs):
        # if request.session['loginuser'] == self.get_object().author:
            return super(CommentUpdate, self).dispatch(request, *args, **kwargs)
        # else:
        #     raise PermissionDenied

def delete_comment(request, pk):
    comment = get_object_or_404(Comment, pk=pk)
    post = comment.post
    if request.session['loginuser'] :
        comment.delete()
        return redirect(post.get_absolute_url())
    else:
        raise PermissionDenied


def delete_post(request, pk):
    post = get_object_or_404(Post, pk=pk)

    if request.session['loginuser'] :
        post.delete()
        return redirect('/blog/')
    else:
        raise PermissionDenied

