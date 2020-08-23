from setuptools import setup


setup(
      name='drop_sklearn_transforms',
      version='1.0',
      description='''
            This is a sample python package for encapsulating custom
            tranforms from scikit-learn into Watson Machine Learning
      ''',
      url='https://github.com/monmartins/sklearn_custom',
      author='Ramon Martins',
      author_email='-',
      license='BSD',
      packages=[
            'drop_sklearn_transforms'
      ],
      zip_safe=False
)
