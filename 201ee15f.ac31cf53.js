(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{107:function(e,t,r){"use strict";r.d(t,"a",(function(){return m})),r.d(t,"b",(function(){return f}));var n=r(0),s=r.n(n);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function c(e,t){if(null==e)return{};var r,n,s=function(e,t){if(null==e)return{};var r,n,s={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(s[r]=e[r]);return s}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(s[r]=e[r])}return s}var l=s.a.createContext({}),p=function(e){var t=s.a.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},m=function(e){var t=p(e.components);return s.a.createElement(l.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return s.a.createElement(s.a.Fragment,{},t)}},b=s.a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,o=e.originalType,a=e.parentName,l=c(e,["components","mdxType","originalType","parentName"]),m=p(r),b=n,f=m["".concat(a,".").concat(b)]||m[b]||d[b]||o;return r?s.a.createElement(f,i(i({ref:t},l),{},{components:r})):s.a.createElement(f,i({ref:t},l))}));function f(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var o=r.length,a=new Array(o);a[0]=b;var i={};for(var c in t)hasOwnProperty.call(t,c)&&(i[c]=t[c]);i.originalType=e,i.mdxType="string"==typeof e?e:n,a[1]=i;for(var l=2;l<o;l++)a[l]=r[l];return s.a.createElement.apply(null,a)}return s.a.createElement.apply(null,r)}b.displayName="MDXCreateElement"},81:function(e,t,r){"use strict";r.r(t),r.d(t,"frontMatter",(function(){return a})),r.d(t,"metadata",(function(){return i})),r.d(t,"toc",(function(){return c})),r.d(t,"default",(function(){return p}));var n=r(3),s=r(8),o=(r(0),r(107)),a={id:"processors",title:"Adding a Processor",sidebar_label:"Adding a Processor"},i={unversionedId:"tutorials/processors",id:"tutorials/processors",isDocsHomePage:!1,title:"Adding a Processor",description:"Processors can be thought of as torchvision transforms which transform a sample into a form usable by the model. Each processor takes in a dictionary and returns a dictionary. Processors are initialized as member variables of the dataset and can be used to preprocess samples in the proper format. Here is how processors work in mmf:",source:"@site/docs/tutorials/processors.md",slug:"/tutorials/processors",permalink:"/docs/tutorials/processors",editUrl:"https://github.com/facebookresearch/mmf/edit/master/website/docs/tutorials/processors.md",version:"current",lastUpdatedBy:"Amanpreet Singh",lastUpdatedAt:1594095833,sidebar_label:"Adding a Processor",sidebar:"docs",previous:{title:"Tutorial: Understanding Checkpointing for Pretraining and Finetuning",permalink:"/docs/tutorials/checkpointing"},next:{title:"Large Scale Hyperparameter Sweeps on Slurm",permalink:"/docs/tutorials/slurm"}},c=[{value:"Create a simple Text Processor",id:"create-a-simple-text-processor",children:[]},{value:"Create an Image Processor",id:"create-an-image-processor",children:[]},{value:"Extending an existing processor: Create a fasttext sentence processor",id:"extending-an-existing-processor-create-a-fasttext-sentence-processor",children:[]},{value:"Next Steps",id:"next-steps",children:[]}],l={toc:c};function p(e){var t=e.components,r=Object(s.a)(e,["components"]);return Object(o.b)("wrapper",Object(n.a)({},l,r,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"Processors can be thought of as torchvision transforms which transform a sample into a form usable by the model. Each processor takes in a dictionary and returns a dictionary. Processors are initialized as member variables of the dataset and can be used to preprocess samples in the proper format. Here is how processors work in mmf:"),Object(o.b)("div",{align:"center"},Object(o.b)("img",{width:"80%",src:"https://i.imgur.com/9sZTiUp.gif"})),Object(o.b)("p",null,"For this tutorial, we will create three different types of processors :"),Object(o.b)("ol",null,Object(o.b)("li",{parentName:"ol"},"a simple processor for text,"),Object(o.b)("li",{parentName:"ol"},"a simple processor for images,"),Object(o.b)("li",{parentName:"ol"},"a text processor by extending an existing vocabulary processor in mmf,")),Object(o.b)("h2",{id:"create-a-simple-text-processor"},"Create a simple Text Processor"),Object(o.b)("p",null,"Here we will create a simple processor that takes a sentence and returns a list of stripped word tokens."),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-python"},'\n# registry is needed to register the processor so it is discoverable by MMF\nfrom mmf.common.registry import registry\n# We will inherit the BaseProcessor in MMF\nfrom mmf.datasets.processors import BaseProcessor\n\n@registry.register_processor("simple_processor")\nclass SimpleProccessor(BaseProcessor):\n    def __init__(self, config, *args, **kwargs):\n        return\n\n    # Override the call method\n    def __call__(self, item):\n        text = item[\'text\']\n        text = [t.strip() for t in text.split(" ")]\n        return {"text": text}\n')),Object(o.b)("p",null,"We can add the processor's configuration to a dataset's config and will be available in the dataset class as ",Object(o.b)("inlineCode",{parentName:"p"},"text_processor")," variable:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-yaml"},"dataset_config:\n  vqa2:\n    processors:\n      text_processor:\n        type: simple_processor\n")),Object(o.b)("p",null,"In this manner, processors can be added to any dataset."),Object(o.b)("h2",{id:"create-an-image-processor"},"Create an Image Processor"),Object(o.b)("p",null,"In this section, we will learn how to add an image processor. We will add a processor that converts any grayscale images to 3 channel image."),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-python"},'\nimport torch\n\n# registry is needed to register the processor so it is discoverable by MMF\nfrom mmf.common.registry import registry\n# We will inherit the BaseProcessor in MMF\nfrom mmf.datasets.processors import BaseProcessor\n\n@registry.register_processor("GrayScale")\nclass GrayScale(BaseProcessor):\n    def __init__(self, *args, **kwargs):\n        return\n\n    def __call__(self, item):\n        return self.transform(item["image"])\n\n    def transform(self, x):\n        assert isinstance(x, torch.Tensor)\n        # Handle grayscale, tile 3 times\n        if x.size(0) == 1:\n            x = torch.cat([x] * 3, dim=0)\n        return x\n\n')),Object(o.b)("p",null,"We will add the processor's configuration to the Hateful Memes dataset's config:"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-yaml"},"dataset_config:\n  vqa2:\n    processors:\n      image_processor:\n        type: torchvision_transforms\n        params:\n          transforms:\n            - ToTensor\n            - GrayScale\n")),Object(o.b)("p",null,"The ",Object(o.b)("inlineCode",{parentName:"p"},"torchvision_transforms")," image processor loads the different transform processor like the ",Object(o.b)("inlineCode",{parentName:"p"},"GrayScale")," one we created and composes them together as torchvision transforms. Here we are adding two transforms, first ",Object(o.b)("inlineCode",{parentName:"p"},"ToTensor"),", which is a native torchvision transform to convert the image to a tensor and then the second ",Object(o.b)("inlineCode",{parentName:"p"},"GrayScale")," which will convert a single channel to 3 channel image tensor. So these transforms will be applied to the images when ",Object(o.b)("inlineCode",{parentName:"p"},"image_processor")," is used on an image from the dataset class."),Object(o.b)("h2",{id:"extending-an-existing-processor-create-a-fasttext-sentence-processor"},"Extending an existing processor: Create a fasttext sentence processor"),Object(o.b)("p",null,"A ",Object(o.b)("a",{parentName:"p",href:"https://github.com/facebookresearch/mmf/blob/f11adf0e4a5a28e85239176c44342f6471550e84/mmf/datasets/processors/processors.py#L361"},Object(o.b)("inlineCode",{parentName:"a"},"fasttext"))," processor is available in MMF that returns word embeddings. Here we will create a ",Object(o.b)("inlineCode",{parentName:"p"},"fasttext")," ",Object(o.b)("em",{parentName:"p"},"sentence")," processor hereby extending the ",Object(o.b)("inlineCode",{parentName:"p"},"fasttext")," word processor."),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-python"},'\nimport torch\n\n# registry is needed to register the processor so it is discoverable by MMF\nfrom mmf.common.registry import registry\n# We will inherit the FastText Processor already present in MMF.\n# FastTextProcessor inherits from VocabProecssor\nfrom mmf.datasets.processors import FastTextProcessor\n\n\n# Register the processor so that MMF can discover it\n@registry.register_processor("fasttext_sentence_vector")\nclass FastTextSentenceVectorProcessor(FastTextProcessor):\n   # Override the call method\n   def __call__(self, item):\n       # This function is present in FastTextProcessor class and loads\n       # fasttext bin\n       self._load_fasttext_model(self.model_file)\n       if "text" in item:\n           text = item["text"]\n       elif "tokens" in item:\n           text = " ".join(item["tokens"])\n\n       # Get a sentence vector for sentence and convert it to torch tensor\n       sentence_vector = torch.tensor(\n           self.model.get_sentence_vector(text),\n           dtype=torch.float\n       )\n       # Return back a dict\n       return {\n           "text": sentence_vector\n       }\n\n   # Make dataset builder happy, return a random number\n   def get_vocab_size(self):\n       return None\n')),Object(o.b)("p",null,"For this processor, we can similarly add the configuration to the a dataset's config and will be available in the dataset class as ",Object(o.b)("inlineCode",{parentName:"p"},"text_processor")," :"),Object(o.b)("pre",null,Object(o.b)("code",{parentName:"pre",className:"language-yaml"},"dataset_config:\n  vqa2:\n    processors:\n      text_processor:\n        type: fasttext_sentence_vector\n        params:\n          max_length: null\n          model_file: wiki.en.bin\n")),Object(o.b)("h2",{id:"next-steps"},"Next Steps"),Object(o.b)("p",null,"Learn more about processors in the ",Object(o.b)("a",{parentName:"p",href:"https://mmf.sh/api/lib/datasets/processors.html"},"processors documentation"),"."))}p.isMDXComponent=!0}}]);