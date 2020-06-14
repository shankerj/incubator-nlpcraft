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
import org.apache.nlpcraft.server.ctxword.{NCContextWordResponse, NCContextWordParameter, NCContextWordManager, NCContextWordRequest}
import org.apache.nlpcraft.server.mdo.{NCContextWordConfigMdo => Config}
import org.apache.nlpcraft.server.nlp.enrichers.NCServerEnricher

import scala.collection.Map

object NCContextWordEnricher extends NCServerEnricher {
    private final val MIN_SENTENCE_SCORE = 0.5
    private final val MIN_SENTENCE_FTEXT = 0.5

    private final val MIN_EXAMPLE_SCORE = 1
    private final val MIN_EXAMPLE_FTEXT = 0.1

    private final val LIMIT = 20

    private case class Holder(
        elementId: String,
        stem: String,
        value: String,
        score: Double,
        bertScore: Option[Double] = None,
        ftextScore: Option[Double] = None
    ) {
        override def toString: String = {
            var s = s"ElementId=$elementId, stem=$stem, value=$value, score=$score"

            bertScore match {
                case Some(score) ⇒ s = s"$s, bertScore=$score"
                case None ⇒ // No-op.
            }

            ftextScore match {
                case Some(score) ⇒ s = s"$s, ftextScore=$score"
                case None ⇒ // No-op.
            }

            s
        }
    }

    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { span ⇒
        super.start()
    }

    override def stop(parent: Span): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    private def tryDirect(cfg: Config, toks: Seq[Token], score: Double): Map[Token, Holder] =
        cfg.synonyms.flatMap { case (elemId, syns) ⇒
            toks.flatMap(tok ⇒
                syns.get(tok.stem) match {
                    case Some(value) ⇒ Some(tok → Holder(elemId, tok.stem, value, score))
                    case None ⇒ None
                }
            )
        }

    private def trySentence(cfg: Config, toks: Seq[Token], ns: NCNlpSentence): Map[Token, Holder] = {
        val words = ns.tokens.map(_.origText)

        val suggs: Seq[Seq[NCContextWordResponse]] =
            NCContextWordManager.suggest(
                toks.map(t ⇒ NCContextWordRequest(words, t.index)),
                NCContextWordParameter(limit = LIMIT, totalScore = MIN_SENTENCE_SCORE, ftextScore = MIN_SENTENCE_FTEXT)
            )

        require(toks.size == suggs.size)

        toks.zip(suggs).
            flatMap { case (tok, suggs) ⇒
                suggs.sortBy(-_.totalScore).flatMap(sugg ⇒
                    cfg.contextWords.toStream.flatMap { case (elemId, stems) ⇒
                        if (stems.contains(sugg.stem))
                            Some(
                                tok →
                                    Holder(
                                        elementId = elemId,
                                        stem = sugg.stem,
                                        value = tok.normText,
                                        score = sugg.totalScore,
                                        bertScore = Some(sugg.bertScore),
                                        ftextScore = Some(sugg.ftextScore)
                                    )
                            )
                        else
                            None
                    }).headOption
            }.toMap
    }

    private def tryExamples(cfg: Config, toks: Seq[Token]): Map[Token, Holder] = {
        case class Value(elementId: String, contextWords: Set[String], token: Token, examplesCount: Int)

        val reqs = collection.mutable.ArrayBuffer.empty[(NCContextWordRequest, Value)]

        cfg.examples.foreach { case (elemId, examples) ⇒
            val ctxWords = cfg.contextWords(elemId)

            for ((exampleWords, idxs) ← examples; tok ← toks) {
                val words = substitute(exampleWords, idxs.map(_ → tok.normText).toMap)

                idxs.map(i ⇒ reqs += NCContextWordRequest(words, i) → Value(elemId, ctxWords, tok, examples.size))
            }
        }

        val allSuggs = NCContextWordManager.suggest(
            reqs.map { case (req, _) ⇒ req },
            NCContextWordParameter(limit = LIMIT, totalScore = MIN_EXAMPLE_SCORE, ftextScore = MIN_EXAMPLE_FTEXT)
        )

        require(allSuggs.size == allSuggs.size)

        reqs.map { case (_, value) ⇒ value }.
            zip(allSuggs).
            // TODO:
            flatMap { case (value, suggs) ⇒
                val filtSuggs = suggs.filter(s ⇒ value.contextWords.contains(s.stem))

//                if (filtSuggs.size == value.examplesCount)
                    Some((value.token, value.elementId) → filtSuggs)
//                else
//                    None
            }.
            flatMap {
                case ((tok, elemId), seq) ⇒
                    seq.map(sugg ⇒
                        tok →
                            Holder(
                                elementId = elemId,
                                stem = sugg.stem,
                                value = tok.normText,
                                score = sugg.totalScore,
                                bertScore = Some(sugg.bertScore),
                                ftextScore = Some(sugg.ftextScore)
                            )
                    )
            }.
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
                    val toks = ns.filter(_.pos.startsWith("N"))

                    def logResults(typ: String, m: Map[Token, Holder]): Unit =
                        m.foreach { case (tok, h) ⇒
                            logger.info(
                                s"Token detected [index=${tok.index}, text=${tok.origText}, detected=$typ, data=$h"
                            )
                        }

                    var m = tryDirect(cfg, toks, Integer.MAX_VALUE)

                    logResults("direct", m)

                    def getOther: Seq[Token] = toks.filter(t ⇒ !m.contains(t))

                    if (m.size != toks.size) {
                        val m1 = trySentence(cfg, getOther, ns)

                        logResults("sentence", m1)

                        m ++= m1

                        if (m.size != toks.size) {
                            val m2 = tryExamples(cfg, getOther)

                            logResults("examples", m2)

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
