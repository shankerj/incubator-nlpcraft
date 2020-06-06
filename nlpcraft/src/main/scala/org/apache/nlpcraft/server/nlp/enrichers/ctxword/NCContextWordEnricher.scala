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
import org.apache.nlpcraft.server.mdo.NCContextWordConfigMdo
import org.apache.nlpcraft.server.ctxword.NCContextWordManager
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

    private def tryDirect(cfg: NCContextWordConfigMdo, t: NCNlpSentenceToken, score: Double): Option[Holder] =
        cfg.synonyms.toStream.flatMap { case (elemId, syns) ⇒
            syns.get(t.stem) match {
                case Some(value) ⇒ Some(Holder(elemId, value, score))
                case None ⇒ None
            }
        }.headOption

    private def trySentence(cfg: NCContextWordConfigMdo, t: NCNlpSentenceToken, ns: NCNlpSentence): Option[Holder] = {
        val txt = ns.tokens.map(_.origText).mkString(" ")
        val suggs = NCContextWordManager.suggest(txt, ns.indexOf(t), MIN_SENTENCE_SCORE, LIMIT)

        println()
        println("t="+t.origText + ", idx=" + ns.indexOf(t))
        println("txt="+txt)
        println("suggs="+suggs)

        suggs.sortBy(-_.score).toStream.flatMap(s ⇒
            cfg.contextWords.find { case (_, mets) ⇒ mets.getOrElse(s.stem, 0.0) > MIN_SENTENCE_SCORE } match {
                case Some((elemId, _)) ⇒
                    println(s"!!!wow trySentence for ${t.normText}, suggestion=$s")

                    Some(Holder(elemId, t.normText, s.score))
                case None ⇒
                    println(s"!!!empty for trySentence for ${t.normText}, txt=$txt, idx=${ns.indexOf(t)}, s=$s")
                    None
            }
        ).headOption
    }

    private def tryExamples(cfg: NCContextWordConfigMdo, t: NCNlpSentenceToken): Option[Holder] =
        cfg.examples.toStream.flatMap { case (elemId, examples) ⇒
            val elemMets = cfg.contextWords(elemId)

            var common: Seq[String] = null
            var scores: Seq[Double] = null
            var ok = true

            for ((exampleWords, idxs) ← examples if ok) {
                val txt = substitute(exampleWords, idxs.map(_ → t.normText).toMap)

                for (idx ← idxs if ok) {
                    val suggs =
                        NCContextWordManager.suggest(txt, idx, MIN_EXAMPLE_SCORE, LIMIT).
                            filter(s ⇒ elemMets.contains(s.stem))

                    if (common == null) {
                        common = suggs.map(_.stem)
                        scores = suggs.map(_.score)
                    }
                    else {
                        val seq = suggs.flatMap(s ⇒ if (common.contains(s.stem)) Some(s.stem → s.score) else None)

                        common = seq.map(_._1)
                        scores = seq.map(_._2)
                    }

                    if (common.isEmpty)
                        ok = false
                }
            }

            if (ok) {
                println(s"!!!wow examples for ${t.normText}, common=$common, examples=${examples.keys.map(_.mkString(" ")).mkString("|")}")
                Some(Holder(elemId, t.normText, scores.min))
            }
            else
                None
        }.headOption

    private def substitute(words: Seq[String], subst: Map[Int, String]): String = {
        require(words.size >= subst.size)
        require(subst.keys.forall(i ⇒ i >= 0 && i < words.length))

        words.zipWithIndex.map { case (w, i) ⇒ subst.getOrElse(i, w) }.mkString(" ")
    }

    override def enrich(ns: NCNlpSentence, parent: Span): Unit =
        startScopedSpan("enrich", parent, "srvReqId" → ns.srvReqId, "txt" → ns.text) { _ ⇒
            ns.ctxWordsConfig match {
                case Some(cfg) ⇒
                    val maxScore = cfg.contextWords.values.flatten.map(_._2).max

                    // TODO: other names.
                    ns.filter(_.pos.startsWith("N")).foreach(t ⇒
                        Seq(
                            () ⇒ tryDirect(cfg, t, maxScore * 10),
                            () ⇒ trySentence(cfg, t, ns)
                            // TODO:
                            //,
                            //() ⇒ tryExamples(cfg, t)
                        ).
                            toStream.
                            flatMap(_.apply()).headOption match {
                                case Some(h) ⇒
                                    val tok = ns(ns.indexOf(t))

                                    tok.add(
                                        NCNlpSentenceNote(
                                            Seq(tok.index),
                                            h.elementId,
                                            "value" → h.value,
                                            "score" → h.score
                                        )
                                    )
                                case None ⇒ None
                            }
                    )
                case None ⇒ // No-op.
            }
        }
}
