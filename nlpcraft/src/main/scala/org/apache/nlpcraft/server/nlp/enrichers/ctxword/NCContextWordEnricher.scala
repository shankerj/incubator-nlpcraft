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
import org.apache.nlpcraft.common.nlp.{NCNlpSentence, NCNlpSentenceNote => Note, NCNlpSentenceToken => Token}
import org.apache.nlpcraft.server.ctxword.{NCContextWord, NCContextWordFactor, NCContextWordManager, NCContextWordRequest}
import org.apache.nlpcraft.server.mdo.{NCContextWordConfigMdo => Config}
import org.apache.nlpcraft.server.nlp.enrichers.NCServerEnricher

import scala.collection.Map

object NCContextWordEnricher extends NCServerEnricher {
    // TODO: score
    private final val MIN_SENTENCE_SCORE = 0.3
    private final val MIN_EXAMPLE_SCORE = 1

    private final val MIN_SENTENCE_BERT = 0.3
    private final val MIN_EXAMPLE_BERT = 0.5

    private final val LIMIT = 10

    private case class Holder(elementId: String, value: String, score: Double)

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { span ⇒
        super.start()
    }

    override def stop(parent: Span): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    private def tryDirect(cfg: Config, toks: Seq[Token], score: Double): Map[Token, Holder] =
        cfg.synonyms.flatMap { case (elemId, syns) ⇒
            toks.flatMap(t ⇒
                syns.get(t.stem) match {
                    case Some(value) ⇒ Some(t → Holder(elemId, value, score))
                    case None ⇒ None
                }
            )
        }

    private def trySentence(cfg: Config, toks: Seq[Token], ns: NCNlpSentence): Map[Token, Holder] = {
        val words = ns.tokens.map(_.origText)

        val allSuggs: Seq[Seq[NCContextWord]] =
            NCContextWordManager.suggest(
                toks.map(t ⇒ NCContextWordRequest(words, t.index)),
                NCContextWordFactor(limit = LIMIT, minTotalScore = MIN_SENTENCE_SCORE, minFtextScore = MIN_SENTENCE_BERT)
            )

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

    private def tryExamples(cfg: Config, toks: Seq[Token]): Map[Token, Holder] = {
        case class Value(elementId: String, contextWords: Map[String, Double], token: Token)

        val reqs = collection.mutable.ArrayBuffer.empty[(NCContextWordRequest, Value)]

        cfg.examples.foreach { case (elemId, examples) ⇒
            val contextWords = cfg.contextWords(elemId)

            for ((exampleWords, idxs) ← examples; tok ← toks) {
                val words = substitute(exampleWords, idxs.map(_ → tok.normText).toMap)

                idxs.map(i ⇒ reqs += NCContextWordRequest(words, i) → Value(elemId, contextWords, tok))
            }
        }

        val allSuggs = NCContextWordManager.suggest(
            reqs.map { case (req, _) ⇒ req },
            NCContextWordFactor(limit = LIMIT, minTotalScore = MIN_EXAMPLE_SCORE, minFtextScore = MIN_EXAMPLE_BERT)
        )

        require(allSuggs.size == allSuggs.size)

        reqs.map { case (_, seq) ⇒ seq }.
            zip(allSuggs).
            map { case (value, suggs) ⇒
                (value.token, value.elementId) → suggs.filter(s ⇒ value.contextWords.contains(s.stem))
            }.
            filter { case (_, seq) ⇒ seq.nonEmpty }.
            flatMap { case ((tok, elemId), seq) ⇒ seq.map(p ⇒ tok → Holder(elemId, tok.normText, p.score)) }.
            groupBy { case (tok, _) ⇒ tok }.
            map { case (tok, seq) ⇒ tok → seq.map { case (_, h) ⇒ h }.minBy(-_.score) }
    }

    private def substitute(words: Seq[String], subst: Map[Int, String]): Seq[String] = {
        require(words.size >= subst.size)
        require(subst.keys.forall(i ⇒ i >= 0 && i < words.length))

        words.zipWithIndex.map { case (w, i) ⇒ subst.getOrElse(i, w) }
    }

    override def enrich(ns: NCNlpSentence, parent: Span): Unit =
        startScopedSpan("enrich", parent, "srvReqId" → ns.srvReqId, "txt" → ns.text) { _ ⇒
            ns.ctxWordsConfig match {
                case Some(cfg) ⇒
                    // TODO: other names.
                    val toks = ns.filter(_.pos.startsWith("N"))

                    var m = tryDirect(cfg, toks, cfg.contextWords.values.flatten.map { case (_, score) ⇒ score }.max * 10)

                    logger.info("!direct=" + m)

                    def getOther: Seq[Token] = toks.filter(t ⇒ !m.contains(t))

                    if (m.size != toks.size) {
                        val m1 = trySentence(cfg, getOther, ns)

                        logger.info("!trySentence=" + m1) // TODO: drop print

                        m ++= m1

                        if (m.size != toks.size) {
                            val m2 = tryExamples(cfg, getOther)

                            logger.info("!tryExamples=" + m2) // TODO: drop print

                            m ++= m2

                        }
                    }

                    m.foreach { case (t, h) ⇒
                        t.add(Note(Seq(t.index), h.elementId, "value" → h.value, "score" → h.score))
                    }
                case None ⇒ // No-op.
            }
        }
}
