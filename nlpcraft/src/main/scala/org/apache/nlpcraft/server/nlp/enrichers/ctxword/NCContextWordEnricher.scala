/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.nlpcraft.server.nlp.enrichers.ctxword

import io.opencensus.trace.Span
import org.apache.nlpcraft.common.NCService
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.nlp.{NCNlpSentence, NCNlpSentenceNote, NCNlpSentenceToken}
import org.apache.nlpcraft.server.ctxword.{NCContextRequest, NCContextWord, NCContextWordManager}
import org.apache.nlpcraft.server.mdo.NCContextWordConfigMdo
import org.apache.nlpcraft.server.nlp.enrichers.NCServerEnricher

import scala.collection.Map

object NCContextWordEnricher extends NCServerEnricher {
    @volatile private var url: String = _

    // TODO: score
    private final val MIN_SENTENCE_SCORE = 0.3
    private final val MIN_EXAMPLE_SCORE = 0.85
    private final val LIMIT = 10

    private case class Holder(elementId: String, value: String, score: Double)

    private object Config extends NCConfigurable {
        lazy val url: String = getStringOrElse("nlpcraft.server.ctxword.url", "http://localhost:5000")
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { span ⇒
        url = Config.url

        if (url.last == '/')
            url = url.dropRight(1)

        addTags(span, "url" → url)

        // Tries to access spaCy proxy server.
        // TODO: add health check.

        logger.info(s"Context Word server connected: $url")

        super.start()
    }

    override def stop(parent: Span): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    private def tryDirect(cfg: NCContextWordConfigMdo, toks: Seq[NCNlpSentenceToken], score: Double): Map[NCNlpSentenceToken, Holder] =
        cfg.synonyms.flatMap { case (elemId, syns) ⇒
            toks.flatMap(t ⇒
                syns.get(t.stem) match {
                    case Some(value) ⇒ Some(t → Holder(elemId, value, score))
                    case None ⇒ None
                }
            )
        }

    private def trySentence(
        cfg: NCContextWordConfigMdo,
        toks: Seq[NCNlpSentenceToken],
        ns: NCNlpSentence
    ): Map[NCNlpSentenceToken, Holder] = {
        val txt = ns.tokens.map(_.origText).mkString(" ")

        val allSuggs: Seq[Seq[NCContextWord]] =
            NCContextWordManager.suggest(toks.map(t ⇒ NCContextRequest(txt, t.index)), MIN_SENTENCE_SCORE, LIMIT)

        require(toks.size == allSuggs.size)

        toks.
            zip(allSuggs).
            flatMap { case (t, suggs) ⇒
                suggs.sortBy(-_.score).flatMap(s ⇒
                    cfg.contextWords.find { case (_, mets) ⇒ mets.getOrElse(s.stem, 0.0) > MIN_SENTENCE_SCORE } match {
                        case Some((elemId, _)) ⇒ Some(t → Holder(elemId, t.normText, s.score))
                        case None ⇒ None
                    })
            }.toMap
    }

    private def tryExamples(cfg: NCContextWordConfigMdo, toks: Seq[NCNlpSentenceToken]): Map[NCNlpSentenceToken, Holder] = {
        val reqs = collection.mutable.ArrayBuffer.empty[(NCContextRequest, (String, Map[String, Double], NCNlpSentenceToken))]

        cfg.examples.foreach { case (elemId, examples) ⇒
            val elemMets = cfg.contextWords(elemId)

            for ((exampleWords, idxs) ← examples; t ← toks) {
                val txt = substitute(exampleWords, idxs.map(_ → t.normText).toMap)

                idxs.map(i ⇒ reqs += NCContextRequest(txt, i) → (elemId, elemMets, t))
            }
        }

        val allSuggs: Seq[Seq[NCContextWord]] = NCContextWordManager.suggest(reqs.map(_._1), MIN_EXAMPLE_SCORE, LIMIT)

        require(allSuggs.size == allSuggs.size)

        reqs.map(_._2).
            zip(allSuggs).
            map { case ((elemId, elemMets, t), suggs) ⇒ (t, elemId) → suggs.filter(s ⇒ elemMets.contains(s.stem)) }.
            filter(_._2.nonEmpty).
            flatMap { case ((t, elemId), seq) ⇒ seq.map(p ⇒ t → Holder(elemId, t.normText, p.score)) }.
            groupBy { case (t, _) ⇒ t }.
            map { case (t, seq) ⇒ t → seq.map { case (_, h) ⇒ h }.minBy(-_.score) }
    }

    private def substitute(words: Seq[String], subst: Map[Int, String]): String = {
        require(words.size >= subst.size)
        require(subst.keys.forall(i ⇒ i >= 0 && i < words.length))

        words.zipWithIndex.map { case (w, i) ⇒ subst.getOrElse(i, w) }.mkString(" ")
    }

    override def enrich(ns: NCNlpSentence, parent: Span): Unit =
        startScopedSpan("enrich", parent, "srvReqId" → ns.srvReqId, "txt" → ns.text) { _ ⇒
            ns.ctxWordsConfig match {
                case Some(cfg) ⇒
                    // TODO: other names.
                    val toks = ns.filter(_.pos.startsWith("N"))

                    var m = tryDirect(cfg, toks, cfg.contextWords.values.flatten.map(_._2).max * 10)

                    if (m.size != toks.size) {
                        m ++= trySentence(cfg, toks.filter(t ⇒ !m.contains(t)), ns)

                        if (m.size != toks.size)
                            m ++= tryExamples(cfg, toks.filter(t ⇒ !m.contains(t)))
                    }

                    m.foreach { case (t, h) ⇒
                        t.add(
                            NCNlpSentenceNote(
                                Seq(t.index),
                                h.elementId,
                                "value" → h.value,
                                "score" → h.score
                            )
                        )
                    }
            }
            case None ⇒ // No-op.
        }
}
