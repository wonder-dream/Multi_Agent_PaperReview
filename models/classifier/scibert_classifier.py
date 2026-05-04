"""
科研文本分类器模型定义
基于 SciBERT 预训练模型，支持领域分类、质量分类和多任务联合训练

模型架构:
    SciBERT(encoder) -> [CLS] -> Dropout(0.1) -> Linear -> 分类输出

支持三种模式:
    1. Domain分类: NLP, CV, ML, AI (4类单标签分类)
    2. Quality分类: accept, reject (2类单标签分类，支持类别不平衡处理)
    3. MultiTask: 共享SciBERT编码器，同时输出domain和quality预测
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple


class SciBERTDomainClassifier(nn.Module):
    """
    科研论文领域分类器
    
    基于SciBERT编码，对输入论文进行四领域分类:
        - NLP (cs.CL): 自然语言处理
        - CV (cs.CV): 计算机视觉
        - ML (cs.LG): 机器学习
        - AI (cs.AI): 人工智能
    
    Args:
        model_name: SciBERT预训练模型名称，默认 allenai/scibert_scivocab_uncased
        num_labels: 领域类别数，默认 4
        dropout_rate: Dropout比率，默认 0.1
        freeze_bert_layers: 是否冻结SciBERT前N层，默认 0 (不冻结)
    
    Example:
        >>> model = SciBERTDomainClassifier()
        >>> outputs = model(input_ids, attention_mask)
        >>> print(outputs.keys())  # dict_keys(['logits', 'probs', 'loss'])
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 4,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        
        # SciBERT 编码器
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size  # 768 for SciBERT
        
        # 可选：冻结部分BERT层以加速训练
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        # 分类头: [CLS] -> Dropout -> Linear
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 初始化分类头权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def _freeze_bert_layers(self, n_layers: int):
        """冻结SciBERT前n层，只训练顶层和分类头"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[Model] 冻结了 SciBERT 的前 {n_layers} 层")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: token IDs, shape (batch_size, seq_len)
            attention_mask: 注意力掩码, shape (batch_size, seq_len)
            labels: 类别标签, shape (batch_size,)，可选
        
        Returns:
            字典包含:
                - logits: (batch_size, num_labels)
                - probs: softmax概率 (batch_size, num_labels)
                - loss: 交叉熵损失 (当labels提供时)
        """
        # SciBERT编码: 取最后一层隐藏状态
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取[CLS]向量作为论文表示 (batch_size, hidden_size)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 分类头
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probs": probs
        }

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result


class SciBERTQualityClassifier(nn.Module):
    """
    科研论文质量分类器
    
    基于SciBERT编码，对论文进行二分类 (accept / reject)
    注意类别不平衡问题 (accept通常占70-80%)，支持class_weight
    
    Args:
        model_name: SciBERT预训练模型名称
        num_labels: 类别数，默认 2 (accept, reject)
        dropout_rate: Dropout比率
        freeze_bert_layers: 冻结层数
        class_weights: 类别权重用于处理不平衡，shape (2,)
    
    Example:
        >>> weights = torch.tensor([1.0, 2.5])  # reject类权重更高
        >>> model = SciBERTQualityClassifier(class_weights=weights)
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 3,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        
        # SciBERT 编码器
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # 类别权重 (用于处理accept/reject不平衡)
        self.register_buffer('class_weights', class_weights)
    
    def _freeze_bert_layers(self, n_layers: int):
        """冻结SciBERT前n层"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[Model] 冻结了 SciBERT 的前 {n_layers} 层")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: token IDs, (batch_size, seq_len)
            attention_mask: 注意力掩码, (batch_size, seq_len)
            labels: 类别标签, (batch_size,)
        
        Returns:
            字典包含 logits, probs, loss
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probs": probs
        }
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
            result["loss"] = loss
        
        return result


class SciBERTMethodTypeClassifier(nn.Module):
    """科研论文方法类型分类器 (Empirical, Theoretical, Survey, Benchmark)"""

    LABELS = ["Empirical", "Theoretical", "Survey", "Benchmark"]

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 4,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _freeze_bert_layers(self, n_layers: int):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[Model] 冻结了 SciBERT 的前 {n_layers} 层")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.softmax(logits, dim=-1)

        result = {"logits": logits, "probs": probs}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss
        return result


class SciBERTMultiTaskClassifier(nn.Module):
    """
    多任务联合分类器

    共享SciBERT编码器，同时训练:
        - Domain分类头: 4类多标签 (NLP, CV, ML, AI)
        - Quality分类头: 3类单标签 (Acceptable, Borderline, Weak Reject)
        - Method分类头: 4类单标签 (Empirical, Theoretical, Survey, Benchmark)
    
    通过共享编码器，两个任务可以互相促进:
        - Domain信息帮助判断质量预期 (不同领域的质量标准不同)
        - Quality信号反馈帮助学习更好的领域表示
    
    Args:
        model_name: SciBERT预训练模型名称
        num_domain_labels: 领域类别数，默认 4
        num_quality_labels: 质量类别数，默认 2
        dropout_rate: Dropout比率
        freeze_bert_layers: 冻结层数
        domain_class_weights: Domain任务类别权重
        quality_class_weights: Quality任务类别权重
        task_weights: 两个任务的损失权重，默认 [1.0, 1.0]
    
    Example:
        >>> model = SciBERTMultiTaskClassifier(
        ...     quality_class_weights=torch.tensor([1.0, 2.5]),
        ...     task_weights=[1.0, 0.8]  # domain权重略高于quality
        ... )
        >>> outputs = model(input_ids, attention_mask, domain_labels, quality_labels)
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_domain_labels: int = 4,
        num_quality_labels: int = 3,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0,
        domain_class_weights: Optional[torch.Tensor] = None,
        quality_class_weights: Optional[torch.Tensor] = None,
        method_class_weights: Optional[torch.Tensor] = None,
        task_weights: list = [1.0, 1.0, 1.0]
    ):
        super().__init__()
        self.num_domain_labels = num_domain_labels
        self.num_quality_labels = num_quality_labels
        self.num_method_labels = getattr(self, 'num_method_labels', 4)
        self.task_weights = task_weights
        self.config = AutoConfig.from_pretrained(model_name)

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        self.dropout = nn.Dropout(dropout_rate)

        # Domain分类头 (多标签)
        self.domain_classifier = nn.Linear(hidden_size, num_domain_labels)
        nn.init.xavier_uniform_(self.domain_classifier.weight)
        nn.init.zeros_(self.domain_classifier.bias)

        # Quality分类头
        self.quality_classifier = nn.Linear(hidden_size, num_quality_labels)
        nn.init.xavier_uniform_(self.quality_classifier.weight)
        nn.init.zeros_(self.quality_classifier.bias)

        # Method type分类头
        self.method_classifier = nn.Linear(hidden_size, self.num_method_labels)
        nn.init.xavier_uniform_(self.method_classifier.weight)
        nn.init.zeros_(self.method_classifier.bias)

        self.register_buffer('domain_class_weights', domain_class_weights)
        self.register_buffer('quality_class_weights', quality_class_weights)
        self.register_buffer('method_class_weights', method_class_weights)
    
    def _freeze_bert_layers(self, n_layers: int):
        """冻结SciBERT前n层"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[Model] 冻结了 SciBERT 的前 {n_layers} 层")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        quality_labels: Optional[torch.Tensor] = None,
        method_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 (三任务: domain多标签 + quality单标签 + method单标签)
        """
        # 共享编码器
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Domain (多标签)
        domain_logits = self.domain_classifier(pooled_output)
        domain_probs = torch.sigmoid(domain_logits)

        # Quality
        quality_logits = self.quality_classifier(pooled_output)
        quality_probs = torch.softmax(quality_logits, dim=-1)

        # Method type
        method_logits = self.method_classifier(pooled_output)
        method_probs = torch.softmax(method_logits, dim=-1)

        result = {
            "domain_logits": domain_logits,
            "domain_probs": domain_probs,
            "quality_logits": quality_logits,
            "quality_probs": quality_probs,
            "method_logits": method_logits,
            "method_probs": method_probs
        }

        total_loss = 0.0

        if domain_labels is not None:
            valid = domain_labels.sum(dim=-1) >= 0
            if valid.any():
                domain_loss_fct = nn.BCEWithLogitsLoss(weight=self.domain_class_weights)
                domain_loss = domain_loss_fct(domain_logits[valid], domain_labels[valid])
                result["domain_loss"] = domain_loss
                total_loss += self.task_weights[0] * domain_loss

        if quality_labels is not None:
            valid = quality_labels >= 0
            if valid.any():
                quality_loss_fct = nn.CrossEntropyLoss(weight=self.quality_class_weights)
                quality_loss = quality_loss_fct(quality_logits[valid], quality_labels[valid])
                result["quality_loss"] = quality_loss
                total_loss += self.task_weights[1] * quality_loss

        if method_labels is not None:
            valid = method_labels >= 0
            if valid.any():
                method_loss_fct = nn.CrossEntropyLoss(weight=self.method_class_weights)
                method_loss = method_loss_fct(method_logits[valid], method_labels[valid])
                result["method_loss"] = method_loss
                total_loss += self.task_weights[2] * method_loss

        if domain_labels is not None or quality_labels is not None or method_labels is not None:
            result["loss"] = total_loss

        return result


class PaperClassifier:
    """
    论文分类器封装类 (供Agent层直接调用)
    
    提供简洁的接口，输入论文文本，输出结构化分类结果。
    内部处理tokenization、模型推理、标签映射等全部流程。
    
    Args:
        model_path: 训练好的模型检查点路径
        model_type: 模型类型，'domain' | 'quality' | 'multitask'
        device: 运行设备
    
    Example:
        >>> classifier = PaperClassifier("checkpoints/domain_best.pt", "domain")
        >>> result = classifier.classify("Title: BERT... Abstract: We introduce...")
        >>> print(result)
        {
            'domains': ['NLP'],
            'method_type': 'Empirical',
            'quality_tier': 'Acceptable',
            'confidence': {'domain': 0.95, 'quality': 0.88}
        }
    """
    
    DOMAIN_LABELS = ["NLP", "CV", "ML", "AI"]
    QUALITY_LABELS = ["Acceptable", "Borderline", "Weak Reject"]
    METHOD_LABELS = ["Empirical", "Theoretical", "Survey", "Benchmark"]
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "multitask",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_type = model_type
        
        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        
        # 加载模型
        if model_type == "domain":
            self.model = SciBERTDomainClassifier()
        elif model_type == "quality":
            self.model = SciBERTQualityClassifier()
        elif model_type == "multitask":
            self.model = SciBERTMultiTaskClassifier()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        print(f"[PaperClassifier] 已加载 {model_type} 模型 from {model_path}")
    
    @torch.no_grad()
    def classify(self, paper_text: str) -> Dict:
        """
        对论文进行分类
        
        Args:
            paper_text: 格式为 "Title: xxx Abstract: xxx" 的论文文本
        
        Returns:
            结构化分类结果字典
        """
        # Tokenize
        encoding = self.tokenizer(
            paper_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # 推理
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        result = {
            "domains": [],
            "method_type": "Empirical",  # 默认假设，method_type需要额外训练数据
            "quality_tier": "Acceptable",
            "confidence": {}
        }
        
        if self.model_type == "domain":
            probs = outputs["probs"][0].cpu().numpy()
            active = [self.DOMAIN_LABELS[i] for i, p in enumerate(probs) if p >= 0.5]
            result["domains"] = active or [self.DOMAIN_LABELS[probs.argmax()]]
            result["confidence"]["domain"] = {l: float(p) for l, p in zip(self.DOMAIN_LABELS, probs)}

        elif self.model_type == "quality":
            probs = outputs["probs"][0].cpu().numpy()
            pred_idx = probs.argmax()
            quality_label = self.QUALITY_LABELS[pred_idx]
            result["quality_tier"] = quality_label
            result["confidence"]["quality"] = float(probs[pred_idx])

        elif self.model_type == "multitask":
            domain_probs = outputs["domain_probs"][0].cpu().numpy()
            quality_probs = outputs["quality_probs"][0].cpu().numpy()
            method_probs = outputs["method_probs"][0].cpu().numpy()

            active = [self.DOMAIN_LABELS[i] for i, p in enumerate(domain_probs) if p >= 0.5]
            result["domains"] = active or [self.DOMAIN_LABELS[domain_probs.argmax()]]
            result["quality_tier"] = self.QUALITY_LABELS[quality_probs.argmax()]
            result["method_type"] = self.METHOD_LABELS[method_probs.argmax()]
            result["confidence"] = {
                "domain": {l: float(p) for l, p in zip(self.DOMAIN_LABELS, domain_probs)},
                "quality": float(quality_probs[quality_probs.argmax()]),
                "method": float(method_probs.max())
            }
        
        return result
    
    @torch.no_grad()
    def classify_batch(self, paper_texts: list) -> list:
        """
        批量分类
        
        Args:
            paper_texts: 论文文本列表
        
        Returns:
            分类结果列表
        """
        encodings = self.tokenizer(
            paper_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        results = []
        batch_size = len(paper_texts)
        
        for i in range(batch_size):
            result = {
                "domains": [],
                "method_type": "Empirical",
                "quality_tier": "Acceptable",
                "confidence": {}
            }
            
            if self.model_type in ["domain", "multitask"]:
                domain_probs = outputs["domain_probs"][i].cpu().numpy()
                active = [self.DOMAIN_LABELS[j] for j, p in enumerate(domain_probs) if p >= 0.5]
                result["domains"] = active or [self.DOMAIN_LABELS[domain_probs.argmax()]]
                result["confidence"]["domain"] = {l: float(p) for l, p in zip(self.DOMAIN_LABELS, domain_probs)}

            if self.model_type in ["quality", "multitask"]:
                quality_probs = outputs["quality_probs"][i].cpu().numpy()
                quality_pred = quality_probs.argmax()
                result["quality_tier"] = self.QUALITY_LABELS[quality_pred]
                result["confidence"]["quality"] = float(quality_probs[quality_pred])

            if self.model_type == "multitask":
                method_probs = outputs["method_probs"][i].cpu().numpy()
                result["method_type"] = self.METHOD_LABELS[method_probs.argmax()]
                result["confidence"]["method"] = float(method_probs.max())

            results.append(result)
        
        return results


def classify_paper(paper_text: str, classifier: "PaperClassifier") -> Dict:
    """
    Agent调用接口: 对论文进行多维度分类
    
    这是给Agent层调用的标准接口函数。
    
    Args:
        paper_text: 论文文本 (Title + Abstract)
        classifier: 已初始化的PaperClassifier实例
    
    Returns:
        结构化分类结果:
        {
            "domains": ["NLP"],
            "method_type": "Empirical",
            "quality_tier": "Acceptable"
        }
    """
    return classifier.classify(paper_text)
