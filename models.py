from pydantic import BaseModel
from pydantic import Field, AliasChoices, AliasPath
# from pydantic.dataclasses import dataclass
from dataclasses import field, dataclass
from typing import List, Dict, Optional


@dataclass
class CustomArticle:
    user_id: int = Field(default=0, validation_alias=AliasChoices('user_id', 'author_id', 'user'))
    src: str = Field(default='', validation_alias=AliasChoices('src',
                                                               AliasPath('source', 'name'),
                                                               'source',
                                                               'author'))
    title: str = ...
    content: str = Field(validation_alias=AliasChoices('content', 'text', 'summary'))
    categories: List[int] | None = field(default_factory=list)
    url: str = Field(validation_alias=AliasChoices('src_url',
                                                   'link',
                                                   'url',
                                                   AliasPath('source', 'url'))) 


def get_test_custom_article(title: str) -> CustomArticle:
    """
    Factory function to create a test CustomArticle instance.
    """
    return CustomArticle(
                         user_id=0,
                         src="",
                         title=title,
                         content="This is a test article content.",
                         categories=["1", "2"],
                         url="")