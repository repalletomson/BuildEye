from django import forms


class ImageComparisonForm(forms.Form):
    initial_image = forms.ImageField(required=True)
    current_image = forms.ImageField(required=True)
    